#!/usr/bin/env python3
"""
Benchmark runtime of the segmentation and keypoint-detection pipelines.

The script is designed for GPU-cluster use and benchmarks both approaches
without writing intermediate predictions to disk. It measures per-image time
for:
  - preprocess
  - model inference
  - postprocessing
  - clinical metric computation
  - total end-to-end time

Output:
  - per_image_timings.csv
  - summary_by_method.csv
  - run_metadata.json

Example:
  python Src/Metrics_and_visualization/benchmark_methods_runtime.py ^
    --methods both ^
    --device cuda ^
    --output_dir C:\\Dev\\Digitech\\runtime_benchmark ^
    --seg_model_path C:\\Dev\\Digitech\\Src\\Models\\Atlas-heqv-multi-100-fold-4-final.pth ^
    --seg_input_dir C:\\cluster\\data\\atlas\\images ^
    --keypoint_repo C:\\Dev\\KeypointDetection ^
    --keypoint_experiment keypoint_inference ^
    --keypoint_checkpoint epoch_500.pt ^
    --keypoint_images_list atlas_verteba_full.csv ^
    --warmup 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Src.Atlas import extraction as atlas_extraction
from Src.Atlas import inference as atlas_inference
from Src.Metrics_and_visualization.compare_datasets_metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the runtime benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark runtime of segmentation and keypoint-detection methods."
    )
    parser.add_argument(
        "--methods",
        choices=["segmentation", "keypoint", "both"],
        default="both",
        help="Which methods to benchmark.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda, cuda:0, cpu.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of initial images to run as warmup per method.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of benchmarked images per method after warmup. 0 = all.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where benchmark outputs will be written.",
    )
    parser.add_argument(
        "--common_files_csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with a common file list. If omitted and both methods are "
            "benchmarked, segmentation is automatically filtered to the filenames "
            "loaded by the keypoint dataset."
        ),
    )

    parser.add_argument("--seg_model_path", type=Path, default=None, help="Segmentation .pth checkpoint.")
    parser.add_argument("--seg_input_dir", type=Path, default=None, help="Directory with PNG images for segmentation.")
    parser.add_argument("--seg_file_glob", default="*.png", help="Glob for segmentation input images.")
    parser.add_argument("--seg_sw_batch_size", type=int, default=4, help="Sliding-window batch size.")
    parser.add_argument("--seg_overlap", type=float, default=0.25, help="Sliding-window overlap.")
    parser.add_argument("--seg_conf_thresh", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--seg_min_size", type=int, default=500, help="Minimum mask component size.")
    parser.add_argument(
        "--seg_no_relabel",
        action="store_true",
        help="Disable anatomical relabeling of vertebra classes after postprocessing.",
    )
    parser.add_argument(
        "--seg_extract_min_area",
        type=int,
        default=300,
        help="Minimum area for vertebra region during point extraction.",
    )

    parser.add_argument(
        "--keypoint_repo",
        type=Path,
        default=None,
        help="Root directory of the KeypointDetection project.",
    )
    parser.add_argument(
        "--keypoint_config_stack",
        default="configs/config_list.yaml",
        help="Path to config stack relative to keypoint repo, or absolute path.",
    )
    parser.add_argument(
        "--keypoint_experiment",
        default=None,
        help="Optional override of experiment name used under results/<experiment>.",
    )
    parser.add_argument(
        "--keypoint_checkpoint",
        default=None,
        help="Optional override of checkpoint filename, e.g. epoch_500.pt.",
    )
    parser.add_argument(
        "--keypoint_images_list",
        default=None,
        help=(
            "Optional override of the CSV file listing input images. Interpreted "
            "relative to keypoint root_dir unless absolute."
        ),
    )

    return parser.parse_args()


def sync_device(device: torch.device) -> None:
    """Synchronize the active CUDA device before timing sensitive operations."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_ms(start: float, device: torch.device) -> float:
    """Return elapsed wall-clock time in milliseconds after device sync."""
    sync_device(device)
    return (time.perf_counter() - start) * 1000.0


def ensure_dir(path: Path) -> None:
    """Create a directory tree when it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def merge_configs(base_cfg: dict[str, Any], override_cfg: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an override configuration into a base configuration."""
    for key, value in override_cfg.items():
        if key in base_cfg and isinstance(base_cfg[key], dict) and isinstance(value, dict):
            merge_configs(base_cfg[key], value)
        else:
            base_cfg[key] = value
    return base_cfg


def resolve_config_path(repo_root: Path, config_path: str) -> Path:
    """Resolve a configuration path relative to the keypoint repository root."""
    path = Path(config_path)
    return path if path.is_absolute() else repo_root / path


def load_stacked_yaml_config(repo_root: Path, stack_path: str) -> dict[str, Any]:
    """Load and merge the layered YAML configuration stack used by the keypoint model."""
    stack_file = resolve_config_path(repo_root, stack_path)
    with stack_file.open("r", encoding="utf-8") as f:
        meta = yaml.safe_load(f) or {}
    config_paths = meta.get("configs", [])
    if not config_paths:
        raise ValueError(f"Config stack {stack_file} does not contain any configs.")

    merged: dict[str, Any] | None = None
    for rel_path in config_paths:
        cfg_path = resolve_config_path(repo_root, rel_path)
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if merged is None:
            merged = cfg
        else:
            merge_configs(merged, cfg)
    return merged or {}


def load_filename_filter(csv_path: Path | None) -> set[str] | None:
    """Load an optional filename whitelist from a CSV file."""
    if csv_path is None:
        return None
    if not csv_path.exists():
        raise FileNotFoundError(f"Filename list CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        return set()

    for candidate in ("filename", "file", "image", "imagePath"):
        if candidate in df.columns:
            series = df[candidate]
            return {str(v).strip() for v in series.dropna().tolist()}

    first_col = df.columns[0]
    return {str(v).strip() for v in df[first_col].dropna().tolist()}


def list_segmentation_images(input_dir: Path, pattern: str, allowed_filenames: set[str] | None) -> list[Path]:
    """List segmentation input images, optionally filtered by filename."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Segmentation input directory not found: {input_dir}")

    paths = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if allowed_filenames is None:
        return paths

    allowed = set(allowed_filenames)
    return [p for p in paths if p.name in allowed]


def extract_points_from_class_mask(mask: np.ndarray, min_area: int) -> dict[str, tuple[float, float]]:
    """Extract ordered anatomical landmark points from a segmentation class mask."""
    if mask.ndim != 2:
        raise ValueError("Expected 2D class mask for point extraction.")

    vertebra_items: list[dict[str, Any]] = []
    for cls in sorted(int(v) for v in np.unique(mask) if int(v) > 0):
        binary = (mask == cls).astype(np.uint8) * 255
        if cv2.countNonZero(binary) < min_area:
            continue
        ctr = atlas_extraction.centroid_of_mask(binary)
        if ctr is None:
            continue
        vertebra_items.append({"class_id": cls, "mask": binary, "centroid": ctr})

    if not vertebra_items:
        raise ValueError("No usable vertebra regions found in segmentation mask.")

    vertebra_items.sort(key=lambda item: item["centroid"][1])
    default_names = ["C2", "C3", "C4", "C5", "C6", "C7"]
    names = default_names[: len(vertebra_items)]

    points: dict[str, tuple[float, float]] = {}
    prev_bl: np.ndarray | None = None
    prev_br: np.ndarray | None = None

    for item, name in zip(vertebra_items, names):
        binary = item["mask"]
        contour = atlas_extraction.main_contour(binary)
        if contour is None or cv2.contourArea(contour) < min_area:
            continue

        if name == "C2":
            tri = atlas_extraction.approx_poly_n(contour, 3)
            if tri is not None:
                tri = atlas_extraction.order_triangle_bl_br_apex(tri)
                tri = atlas_extraction.refine_subpix(binary, tri, win=7)
                tri = atlas_extraction.order_triangle_bl_br_apex(tri)
                bl, br, apex_ref = tri
            else:
                bl, br = atlas_extraction.bottom_edge_endpoints_from_contour(
                    contour, band_frac=0.5, min_band_px=8
                )
                apex = atlas_extraction.apex_top_point_from_contour(contour, y_tol=3)
                pts3 = np.stack([bl, br, apex], axis=0)
                pts3 = atlas_extraction.refine_subpix(binary, pts3, win=7)
                bl, br, apex_ref = pts3[0], pts3[1], pts3[2]

            points["C2 bottom left"] = (float(bl[0]), float(bl[1]))
            points["C2 bottom right"] = (float(br[0]), float(br[1]))
            points["C2 centroid"] = (float(apex_ref[0]), float(apex_ref[1]))
            prev_bl = bl.astype(np.float32)
            prev_br = br.astype(np.float32)
            continue

        quad = atlas_extraction.approx_quad(contour)
        if quad is None:
            continue
        quad = atlas_extraction.refine_subpix(binary, quad, win=7)

        if not atlas_extraction.quad_is_valid(
            quad, min_pair_dist=5.0, min_area=30.0, min_edge=4.0
        ):
            rect = cv2.minAreaRect(cv2.convexHull(contour))
            quad = cv2.boxPoints(rect).astype(np.float32)

        quad = atlas_extraction.order_quad_from_previous_bottom(quad, prev_bl, prev_br)
        tl, tr, br, bl = quad

        points[f"{name} top left"] = (float(tl[0]), float(tl[1]))
        points[f"{name} top right"] = (float(tr[0]), float(tr[1]))
        points[f"{name} bottom right"] = (float(br[0]), float(br[1]))
        points[f"{name} bottom left"] = (float(bl[0]), float(bl[1]))

        prev_bl = bl.astype(np.float32)
        prev_br = br.astype(np.float32)

    return points


def summarize_column(values: pd.Series) -> dict[str, float]:
    """Summarize one timing column with common descriptive statistics."""
    arr = values.dropna().astype(float).to_numpy()
    if arr.size == 0:
        return {
            "n": 0,
            "mean_ms": math.nan,
            "median_ms": math.nan,
            "std_ms": math.nan,
            "min_ms": math.nan,
            "max_ms": math.nan,
            "p95_ms": math.nan,
            "fps_from_mean": math.nan,
        }
    mean_ms = float(np.mean(arr))
    return {
        "n": int(arr.size),
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "std_ms": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps_from_mean": float(1000.0 / mean_ms) if mean_ms > 0 else math.nan,
    }


def build_summary(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image timing rows into method-level summary statistics."""
    stages = [
        "preprocess_ms",
        "model_inference_ms",
        "postprocess_ms",
        "metric_compute_ms",
        "total_ms",
    ]
    rows: list[dict[str, Any]] = []

    for method, method_df in per_image_df.groupby("method"):
        metric_success = int(method_df["metric_status"].eq("ok").sum())
        total_images = int(len(method_df))
        for stage in stages:
            stats = summarize_column(method_df[stage])
            row = {
                "method": method,
                "stage": stage,
                "n_images": total_images,
                "metric_success_count": metric_success,
                "metric_success_ratio": (metric_success / total_images) if total_images else math.nan,
            }
            row.update(stats)
            rows.append(row)

    return pd.DataFrame(rows)


def benchmark_segmentation_setup(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Load segmentation model assets needed for repeated runtime measurements."""
    if args.seg_model_path is None or args.seg_input_dir is None:
        raise ValueError("Segmentation benchmark requires --seg_model_path and --seg_input_dir.")

    model = atlas_inference.load_trained_model(str(args.seg_model_path), device)
    transform = atlas_inference.get_inference_transform_sw(roi_size=atlas_inference.ROI_SIZE)
    return {
        "model": model,
        "transform": transform,
        "device": device,
        "apply_relabel": not args.seg_no_relabel,
    }


def benchmark_segmentation_one(
    image_path: Path,
    setup: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Benchmark one full segmentation pipeline pass for a single image."""
    device: torch.device = setup["device"]
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    img_pil = Image.open(image_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)
    orig_h, orig_w = img_np_hw3.shape[:2]
    img_np = np.transpose(img_np_hw3, (2, 0, 1))
    data = {"image": img_np, "image_raw": img_np.copy()}
    data = setup["transform"](data)
    img_tensor = data["image"].unsqueeze(0).to(device)
    preprocess_ms = elapsed_ms(preprocess_start, device)

    model_start = time.perf_counter()
    logits = atlas_inference.sliding_window_inference(
        inputs=img_tensor,
        roi_size=atlas_inference.ROI_SIZE,
        sw_batch_size=args.seg_sw_batch_size,
        predictor=setup["model"],
        overlap=args.seg_overlap,
        mode="gaussian",
        device=device,
        sw_device=device,
    )
    model_inference_ms = elapsed_ms(model_start, device)

    post_start = time.perf_counter()
    probs = torch.softmax(logits, dim=1)
    max_prob, pred = probs.max(dim=1)
    pred_mask = pred.squeeze(0).detach().cpu().numpy()
    max_prob_np = max_prob.squeeze(0).detach().cpu().numpy()
    pred_mask[max_prob_np < args.seg_conf_thresh] = 0
    cleaned_mask = atlas_inference.postprocess_mask(
        pred_mask,
        num_classes=atlas_inference.NUM_CLASSES,
        min_size=args.seg_min_size,
    )
    if setup["apply_relabel"]:
        cleaned_mask = atlas_inference.relabel_by_vertical_position(cleaned_mask)
    cleaned_mask = cleaned_mask[:orig_h, :orig_w].astype(np.uint8)

    extract_start = time.perf_counter()
    extraction_status = "ok"
    extraction_error = ""
    points: dict[str, tuple[float, float]] | None = None
    try:
        points = extract_points_from_class_mask(cleaned_mask, min_area=args.seg_extract_min_area)
    except Exception as exc:  # noqa: BLE001
        extraction_status = "failed"
        extraction_error = str(exc)
    postprocess_ms = elapsed_ms(post_start, device)
    point_extraction_ms = elapsed_ms(extract_start, device)

    metric_start = time.perf_counter()
    metric_status = "skipped"
    metric_error = ""
    try:
        if points is None:
            raise ValueError(extraction_error or "Point extraction did not produce output.")
        _ = compute_metrics(points)
        metric_status = "ok"
    except Exception as exc:  # noqa: BLE001
        metric_status = "failed"
        metric_error = str(exc)
    metric_compute_ms = elapsed_ms(metric_start, device)
    total_ms = elapsed_ms(total_start, device)

    return {
        "method": "segmentation",
        "filename": image_path.name,
        "preprocess_ms": preprocess_ms,
        "model_inference_ms": model_inference_ms,
        "postprocess_ms": postprocess_ms,
        "metric_compute_ms": metric_compute_ms,
        "total_ms": total_ms,
        "point_extraction_ms": point_extraction_ms,
        "metric_status": metric_status,
        "metric_error": metric_error,
        "postprocess_status": extraction_status,
        "postprocess_error": extraction_error,
    }


def load_keypoint_labels(label_path: Path) -> list[str]:
    """Load the ordered keypoint label names used by the keypoint detector."""
    with label_path.open("r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if not labels:
        raise ValueError(f"No keypoint labels found in {label_path}")
    return labels


def decode_heatmaps_to_points(
    heatmaps: np.ndarray,
    labels: list[str],
    target_size: tuple[int, int],
) -> dict[str, tuple[float, float]]:
    """Convert keypoint heatmaps into image-space point coordinates."""
    if heatmaps.ndim != 3:
        raise ValueError("Expected heatmaps with shape (K, H, W).")

    target_w, target_h = target_size
    heat_h = heatmaps.shape[1]
    heat_w = heatmaps.shape[2]
    sx = target_w / float(heat_w)
    sy = target_h / float(heat_h)

    points: dict[str, tuple[float, float]] = {}
    for idx, label in enumerate(labels):
        heatmap = heatmaps[idx]
        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points[label.replace("_", " ")] = (float(x * sx), float(y * sy))
    return points


def load_keypoint_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load a keypoint model checkpoint into an instantiated model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()


def benchmark_keypoint_setup(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """Load the keypoint detection stack and return reusable benchmark state."""
    if args.keypoint_repo is None:
        raise ValueError("Keypoint benchmark requires --keypoint_repo.")

    repo_root = args.keypoint_repo.resolve()
    if not repo_root.exists():
        raise FileNotFoundError(f"Keypoint repo not found: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.datasets.factory import get_dataset  # type: ignore
    from src.models.factory import get_model  # type: ignore

    cfg = load_stacked_yaml_config(repo_root, args.keypoint_config_stack)

    if args.keypoint_experiment is not None:
        cfg.setdefault("experiment", {})["name"] = args.keypoint_experiment
    if args.keypoint_checkpoint is not None:
        cfg.setdefault("predict", {})["model"] = args.keypoint_checkpoint
    if args.keypoint_images_list is not None:
        cfg.setdefault("predict", {})["images_list"] = args.keypoint_images_list

    root_dir = Path(cfg["predict"]["params"]["root_dir"])
    if not root_dir.is_absolute():
        root_dir = repo_root / root_dir
    root_dir = root_dir.resolve()
    cfg["predict"]["params"]["root_dir"] = str(root_dir)

    annotation_label = Path(cfg["predict"]["annotation_label"])
    if not annotation_label.is_absolute():
        annotation_label = root_dir / annotation_label
    annotation_label = annotation_label.resolve()

    checkpoint_name = cfg["predict"]["model"]
    checkpoint_path = repo_root / "results" / cfg["experiment"]["name"] / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Keypoint checkpoint not found: {checkpoint_path}")

    dataset = get_dataset(
        name="images",
        load=cfg["predict"]["images_list"],
        num_samples=cfg["predict"]["num_samples"],
        keypoint_format=None,
        **cfg["predict"]["params"],
    )

    model = get_model(cfg["model"]["name"], **cfg["model"]["params"])
    load_keypoint_checkpoint(model, checkpoint_path, device)
    target_size = tuple(int(v) for v in cfg["predict"]["params"]["input_size"])
    labels = load_keypoint_labels(annotation_label)

    return {
        "repo_root": repo_root,
        "cfg": cfg,
        "dataset": dataset,
        "model": model,
        "labels": labels,
        "target_size": target_size,
        "special_mode": cfg["model"].get("special_mode"),
        "device": device,
    }


def benchmark_keypoint_one(
    sample_index: int,
    setup: dict[str, Any],
) -> dict[str, Any]:
    """Benchmark one full keypoint-detection pipeline pass for a dataset sample."""
    device: torch.device = setup["device"]
    dataset = setup["dataset"]
    total_start = time.perf_counter()

    preprocess_start = time.perf_counter()
    sample = dataset[sample_index]
    image_np = np.expand_dims(sample["image"], axis=0)
    images = torch.as_tensor(image_np, dtype=torch.float32, device=device)
    preprocess_ms = elapsed_ms(preprocess_start, device)

    model_start = time.perf_counter()
    with torch.no_grad():
        preds = setup["model"](images)
        if setup["special_mode"] == "cut_five_dim" and preds.ndim == 5:
            preds = preds[:, -1, :, :, :]
    model_inference_ms = elapsed_ms(model_start, device)

    post_start = time.perf_counter()
    heatmaps = preds[0].detach().cpu().numpy()
    points = decode_heatmaps_to_points(heatmaps, setup["labels"], setup["target_size"])
    postprocess_ms = elapsed_ms(post_start, device)

    metric_start = time.perf_counter()
    metric_status = "skipped"
    metric_error = ""
    try:
        _ = compute_metrics(points)
        metric_status = "ok"
    except Exception as exc:  # noqa: BLE001
        metric_status = "failed"
        metric_error = str(exc)
    metric_compute_ms = elapsed_ms(metric_start, device)
    total_ms = elapsed_ms(total_start, device)

    return {
        "method": "keypoint",
        "filename": str(sample["filename"]),
        "preprocess_ms": preprocess_ms,
        "model_inference_ms": model_inference_ms,
        "postprocess_ms": postprocess_ms,
        "metric_compute_ms": metric_compute_ms,
        "total_ms": total_ms,
        "point_extraction_ms": math.nan,
        "metric_status": metric_status,
        "metric_error": metric_error,
        "postprocess_status": "ok",
        "postprocess_error": "",
    }


def benchmark_with_warmup(items: list[Any], warmup: int, fn) -> list[dict[str, Any]]:
    """Run warmup iterations first and then collect benchmark rows."""
    if warmup > 0:
        for item in items[:warmup]:
            fn(item)

    rows: list[dict[str, Any]] = []
    for item in items[warmup:]:
        rows.append(fn(item))
    return rows


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    """Write benchmark metadata to a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    """Execute the requested runtime benchmark and save all outputs."""
    args = parse_args()
    ensure_dir(args.output_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    methods = ["segmentation", "keypoint"] if args.methods == "both" else [args.methods]
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "device": str(device),
        "warmup": int(args.warmup),
        "limit": int(args.limit),
        "methods": methods,
        "notes": [
            "Disk writes of intermediate predictions are intentionally excluded.",
            "Metric timing measures computation on predicted points only.",
        ],
    }

    common_filenames = load_filename_filter(args.common_files_csv)

    keypoint_setup: dict[str, Any] | None = None
    if "keypoint" in methods:
        keypoint_setup = benchmark_keypoint_setup(args, device)
        keypoint_filenames = list(getattr(keypoint_setup["dataset"], "images"))
        metadata["keypoint_dataset_size"] = len(keypoint_filenames)
        if common_filenames is None and "segmentation" in methods:
            common_filenames = set(keypoint_filenames)

    if "segmentation" in methods:
        seg_setup = benchmark_segmentation_setup(args, device)
        seg_images = list_segmentation_images(args.seg_input_dir, args.seg_file_glob, common_filenames)
        if args.limit > 0:
            seg_items = seg_images[: args.warmup + args.limit]
        else:
            seg_items = seg_images
        metadata["segmentation_image_count"] = len(seg_items)
        rows.extend(benchmark_with_warmup(seg_items, args.warmup, lambda p: benchmark_segmentation_one(p, seg_setup, args)))

    if "keypoint" in methods and keypoint_setup is not None:
        dataset = keypoint_setup["dataset"]
        kp_indices = list(range(len(dataset)))
        if args.limit > 0:
            kp_items = kp_indices[: args.warmup + args.limit]
        else:
            kp_items = kp_indices
        metadata["keypoint_benchmark_count"] = len(kp_items)
        rows.extend(benchmark_with_warmup(kp_items, args.warmup, lambda idx: benchmark_keypoint_one(idx, keypoint_setup)))

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    per_image_df = pd.DataFrame(rows)
    summary_df = build_summary(per_image_df)

    per_image_path = args.output_dir / "per_image_timings.csv"
    summary_path = args.output_dir / "summary_by_method.csv"
    metadata_path = args.output_dir / "run_metadata.json"

    per_image_df.to_csv(per_image_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_metadata(metadata_path, metadata)

    print(f"Saved per-image timings to: {per_image_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
