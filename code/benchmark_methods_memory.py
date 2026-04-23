#!/usr/bin/env python3
"""
Benchmark memory usage of the segmentation and keypoint-detection pipelines.

The script mirrors benchmark_methods_runtime.py, but instead of measuring time
it tracks process CPU memory (working set / RSS) and, when CUDA is used, GPU
allocated/reserved memory. Intermediate predictions are kept in memory and are
not written to disk.

Output:
  - per_image_memory.csv
  - summary_by_method.csv
  - run_metadata.json
"""

from __future__ import annotations

import argparse
import math
import platform
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from Src.Metrics_and_visualization import benchmark_methods_runtime as runtime_bench


MB = 1024.0 * 1024.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the memory benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark memory usage of segmentation and keypoint-detection methods."
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
    parser.add_argument(
        "--monitor_interval_ms",
        type=float,
        default=2.0,
        help="Polling interval for memory sampling in milliseconds.",
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


def get_process_rss_bytes() -> int:
    """Return the current process resident memory size in bytes."""
    system = platform.system()
    if system == "Windows":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            process,
            ctypes.byref(counters),
            counters.cb,
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed.")
        return int(counters.WorkingSetSize)

    if system == "Linux":
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        rss_pages = int(parts[1])
        page_size = int(getattr(__import__("os"), "sysconf")("SC_PAGE_SIZE"))
        return rss_pages * page_size

    raise NotImplementedError(f"Process RSS sampling is not implemented for {system}.")


def get_gpu_memory_bytes(device: torch.device) -> tuple[int | None, int | None]:
    """Return current allocated and reserved GPU memory for the given device."""
    if device.type != "cuda":
        return None, None
    allocated = int(torch.cuda.memory_allocated(device))
    reserved = int(torch.cuda.memory_reserved(device))
    return allocated, reserved


class MemoryMonitor:
    """Continuously sample CPU and GPU memory usage during one benchmark stage."""

    def __init__(self, device: torch.device, interval_s: float) -> None:
        """Initialize monitoring state and peak trackers."""
        self.device = device
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self.cpu_before = 0
        self.cpu_after = 0
        self.cpu_peak = 0

        self.gpu_alloc_before: int | None = None
        self.gpu_alloc_after: int | None = None
        self.gpu_alloc_peak: int | None = None

        self.gpu_reserved_before: int | None = None
        self.gpu_reserved_after: int | None = None
        self.gpu_reserved_peak: int | None = None

    def _sample_once(self) -> None:
        """Capture one instantaneous CPU/GPU memory sample."""
        cpu_now = get_process_rss_bytes()
        self.cpu_peak = max(self.cpu_peak, cpu_now)

        gpu_alloc, gpu_reserved = get_gpu_memory_bytes(self.device)
        if gpu_alloc is not None:
            if self.gpu_alloc_peak is None:
                self.gpu_alloc_peak = gpu_alloc
            else:
                self.gpu_alloc_peak = max(self.gpu_alloc_peak, gpu_alloc)
        if gpu_reserved is not None:
            if self.gpu_reserved_peak is None:
                self.gpu_reserved_peak = gpu_reserved
            else:
                self.gpu_reserved_peak = max(self.gpu_reserved_peak, gpu_reserved)

    def _run(self) -> None:
        """Run the background sampling loop until monitoring is stopped."""
        while not self._stop_event.wait(self.interval_s):
            self._sample_once()

    def start(self) -> None:
        """Start background memory monitoring and record baseline values."""
        runtime_bench.sync_device(self.device)
        self.cpu_before = get_process_rss_bytes()
        self.cpu_peak = self.cpu_before

        self.gpu_alloc_before, self.gpu_reserved_before = get_gpu_memory_bytes(self.device)
        self.gpu_alloc_peak = self.gpu_alloc_before
        self.gpu_reserved_peak = self.gpu_reserved_before

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, float]:
        """Stop monitoring and return normalized delta and peak statistics."""
        runtime_bench.sync_device(self.device)
        self._sample_once()
        self.cpu_after = get_process_rss_bytes()
        self.gpu_alloc_after, self.gpu_reserved_after = get_gpu_memory_bytes(self.device)

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        return {
            "cpu_delta_mb": (self.cpu_after - self.cpu_before) / MB,
            "cpu_peak_increase_mb": max(self.cpu_peak - self.cpu_before, 0) / MB,
            "gpu_allocated_delta_mb": nan_or_mb(self.gpu_alloc_after, self.gpu_alloc_before),
            "gpu_allocated_peak_increase_mb": nan_or_mb(self.gpu_alloc_peak, self.gpu_alloc_before, peak=True),
            "gpu_reserved_delta_mb": nan_or_mb(self.gpu_reserved_after, self.gpu_reserved_before),
            "gpu_reserved_peak_increase_mb": nan_or_mb(self.gpu_reserved_peak, self.gpu_reserved_before, peak=True),
        }


def nan_or_mb(current: int | None, baseline: int | None, peak: bool = False) -> float:
    """Convert a byte delta to megabytes while preserving missing values."""
    if current is None or baseline is None:
        return math.nan
    diff = current - baseline
    if peak:
        diff = max(diff, 0)
    return diff / MB


def measure_memory(device: torch.device, interval_s: float, fn):
    """Run a callable while tracking memory consumption around it."""
    monitor = MemoryMonitor(device=device, interval_s=interval_s)
    monitor.start()
    try:
        result = fn()
    finally:
        stats = monitor.stop()
    return result, stats


def stage_stats_to_record(prefix: str, stats: dict[str, float]) -> dict[str, float]:
    """Prefix one set of stage memory statistics for DataFrame storage."""
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def summarize_column(values: pd.Series) -> dict[str, float]:
    """Summarize one memory column with common descriptive statistics."""
    arr = values.dropna().astype(float).to_numpy()
    if arr.size == 0:
        return {
            "n": 0,
            "mean_mb": math.nan,
            "median_mb": math.nan,
            "std_mb": math.nan,
            "min_mb": math.nan,
            "max_mb": math.nan,
            "p95_mb": math.nan,
        }
    return {
        "n": int(arr.size),
        "mean_mb": float(np.mean(arr)),
        "median_mb": float(np.median(arr)),
        "std_mb": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min_mb": float(np.min(arr)),
        "max_mb": float(np.max(arr)),
        "p95_mb": float(np.percentile(arr, 95)),
    }


def build_summary(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-image memory rows into method-level summary statistics."""
    stages = [
        "preprocess",
        "model_inference",
        "postprocess",
        "point_extraction",
        "metric_compute",
        "total",
    ]
    metrics = [
        "cpu_delta_mb",
        "cpu_peak_increase_mb",
        "gpu_allocated_delta_mb",
        "gpu_allocated_peak_increase_mb",
        "gpu_reserved_delta_mb",
        "gpu_reserved_peak_increase_mb",
    ]

    rows: list[dict[str, Any]] = []
    for method, method_df in per_image_df.groupby("method"):
        metric_success = int(method_df["metric_status"].eq("ok").sum())
        total_images = int(len(method_df))
        for stage in stages:
            for metric in metrics:
                column = f"{stage}_{metric}"
                if column not in method_df.columns:
                    continue
                stats = summarize_column(method_df[column])
                row = {
                    "method": method,
                    "stage": stage,
                    "memory_metric": metric,
                    "n_images": total_images,
                    "metric_success_count": metric_success,
                    "metric_success_ratio": (metric_success / total_images) if total_images else math.nan,
                }
                row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


def benchmark_segmentation_one(
    image_path: Path,
    setup: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Benchmark memory usage of the segmentation pipeline for one image."""
    device: torch.device = setup["device"]
    interval_s = args.monitor_interval_ms / 1000.0

    total_monitor = MemoryMonitor(device=device, interval_s=interval_s)
    total_monitor.start()
    try:
        def preprocess_stage():
            img_pil = Image.open(image_path).convert("RGB")
            img_np_hw3 = np.array(img_pil, dtype=np.float32)
            orig_h, orig_w = img_np_hw3.shape[:2]
            img_np = np.transpose(img_np_hw3, (2, 0, 1))
            data = {"image": img_np, "image_raw": img_np.copy()}
            data = setup["transform"](data)
            img_tensor = data["image"].unsqueeze(0).to(device)
            return img_tensor, orig_h, orig_w

        (img_tensor, orig_h, orig_w), preprocess_stats = measure_memory(device, interval_s, preprocess_stage)

        def model_stage():
            return runtime_bench.atlas_inference.sliding_window_inference(
                inputs=img_tensor,
                roi_size=runtime_bench.atlas_inference.ROI_SIZE,
                sw_batch_size=args.seg_sw_batch_size,
                predictor=setup["model"],
                overlap=args.seg_overlap,
                mode="gaussian",
                device=device,
                sw_device=device,
            )

        logits, model_stats = measure_memory(device, interval_s, model_stage)

        point_extraction_stats = {
            "cpu_delta_mb": math.nan,
            "cpu_peak_increase_mb": math.nan,
            "gpu_allocated_delta_mb": math.nan,
            "gpu_allocated_peak_increase_mb": math.nan,
            "gpu_reserved_delta_mb": math.nan,
            "gpu_reserved_peak_increase_mb": math.nan,
        }

        def postprocess_stage():
            probs = torch.softmax(logits, dim=1)
            max_prob, pred = probs.max(dim=1)
            pred_mask = pred.squeeze(0).detach().cpu().numpy()
            max_prob_np = max_prob.squeeze(0).detach().cpu().numpy()
            pred_mask[max_prob_np < args.seg_conf_thresh] = 0
            cleaned_mask = runtime_bench.atlas_inference.postprocess_mask(
                pred_mask,
                num_classes=runtime_bench.atlas_inference.NUM_CLASSES,
                min_size=args.seg_min_size,
            )
            if setup["apply_relabel"]:
                cleaned_mask = runtime_bench.atlas_inference.relabel_by_vertical_position(cleaned_mask)
            cleaned_mask = cleaned_mask[:orig_h, :orig_w].astype(np.uint8)

            extraction_status = "ok"
            extraction_error = ""
            points: dict[str, tuple[float, float]] | None = None

            def extraction_stage():
                return runtime_bench.extract_points_from_class_mask(
                    cleaned_mask,
                    min_area=args.seg_extract_min_area,
                )

            nonlocal point_extraction_stats
            try:
                points, point_extraction_stats = measure_memory(device, interval_s, extraction_stage)
            except Exception as exc:  # noqa: BLE001
                extraction_status = "failed"
                extraction_error = str(exc)

            return cleaned_mask, points, extraction_status, extraction_error

        (_, points, extraction_status, extraction_error), postprocess_stats = measure_memory(
            device, interval_s, postprocess_stage
        )

        def metric_stage():
            if points is None:
                raise ValueError(extraction_error or "Point extraction did not produce output.")
            return runtime_bench.compute_metrics(points)

        metric_status = "ok"
        metric_error = ""
        try:
            _, metric_stats = measure_memory(device, interval_s, metric_stage)
        except Exception as exc:  # noqa: BLE001
            metric_status = "failed"
            metric_error = str(exc)
            metric_stats = {
                "cpu_delta_mb": math.nan,
                "cpu_peak_increase_mb": math.nan,
                "gpu_allocated_delta_mb": math.nan,
                "gpu_allocated_peak_increase_mb": math.nan,
                "gpu_reserved_delta_mb": math.nan,
                "gpu_reserved_peak_increase_mb": math.nan,
            }

    finally:
        total_stats = total_monitor.stop()

    row = {
        "method": "segmentation",
        "filename": image_path.name,
        "metric_status": metric_status,
        "metric_error": metric_error,
        "postprocess_status": extraction_status,
        "postprocess_error": extraction_error,
    }
    row.update(stage_stats_to_record("preprocess", preprocess_stats))
    row.update(stage_stats_to_record("model_inference", model_stats))
    row.update(stage_stats_to_record("postprocess", postprocess_stats))
    row.update(stage_stats_to_record("point_extraction", point_extraction_stats))
    row.update(stage_stats_to_record("metric_compute", metric_stats))
    row.update(stage_stats_to_record("total", total_stats))
    return row


def benchmark_keypoint_one(
    sample_index: int,
    setup: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Benchmark memory usage of the keypoint pipeline for one sample."""
    device: torch.device = setup["device"]
    interval_s = args.monitor_interval_ms / 1000.0
    dataset = setup["dataset"]

    total_monitor = MemoryMonitor(device=device, interval_s=interval_s)
    total_monitor.start()
    try:
        def preprocess_stage():
            sample = dataset[sample_index]
            image_np = np.expand_dims(sample["image"], axis=0)
            images = torch.as_tensor(image_np, dtype=torch.float32, device=device)
            return sample, images

        (sample, images), preprocess_stats = measure_memory(device, interval_s, preprocess_stage)

        def model_stage():
            with torch.no_grad():
                preds = setup["model"](images)
                if setup["special_mode"] == "cut_five_dim" and preds.ndim == 5:
                    preds = preds[:, -1, :, :, :]
            return preds

        preds, model_stats = measure_memory(device, interval_s, model_stage)

        def postprocess_stage():
            heatmaps = preds[0].detach().cpu().numpy()
            return runtime_bench.decode_heatmaps_to_points(
                heatmaps,
                setup["labels"],
                setup["target_size"],
            )

        points, postprocess_stats = measure_memory(device, interval_s, postprocess_stage)

        point_extraction_stats = {
            "cpu_delta_mb": math.nan,
            "cpu_peak_increase_mb": math.nan,
            "gpu_allocated_delta_mb": math.nan,
            "gpu_allocated_peak_increase_mb": math.nan,
            "gpu_reserved_delta_mb": math.nan,
            "gpu_reserved_peak_increase_mb": math.nan,
        }

        def metric_stage():
            return runtime_bench.compute_metrics(points)

        metric_status = "ok"
        metric_error = ""
        try:
            _, metric_stats = measure_memory(device, interval_s, metric_stage)
        except Exception as exc:  # noqa: BLE001
            metric_status = "failed"
            metric_error = str(exc)
            metric_stats = {
                "cpu_delta_mb": math.nan,
                "cpu_peak_increase_mb": math.nan,
                "gpu_allocated_delta_mb": math.nan,
                "gpu_allocated_peak_increase_mb": math.nan,
                "gpu_reserved_delta_mb": math.nan,
                "gpu_reserved_peak_increase_mb": math.nan,
            }
    finally:
        total_stats = total_monitor.stop()

    row = {
        "method": "keypoint",
        "filename": str(sample["filename"]),
        "metric_status": metric_status,
        "metric_error": metric_error,
        "postprocess_status": "ok",
        "postprocess_error": "",
    }
    row.update(stage_stats_to_record("preprocess", preprocess_stats))
    row.update(stage_stats_to_record("model_inference", model_stats))
    row.update(stage_stats_to_record("postprocess", postprocess_stats))
    row.update(stage_stats_to_record("point_extraction", point_extraction_stats))
    row.update(stage_stats_to_record("metric_compute", metric_stats))
    row.update(stage_stats_to_record("total", total_stats))
    return row


def benchmark_with_warmup(items: list[Any], warmup: int, fn) -> list[dict[str, Any]]:
    """Run warmup iterations first and collect memory rows afterwards."""
    if warmup > 0:
        for item in items[:warmup]:
            fn(item)

    rows: list[dict[str, Any]] = []
    for item in items[warmup:]:
        rows.append(fn(item))
    return rows


def main() -> None:
    """Execute the requested memory benchmark and save all outputs."""
    args = parse_args()
    runtime_bench.ensure_dir(args.output_dir)

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
        "monitor_interval_ms": float(args.monitor_interval_ms),
        "notes": [
            "Disk writes of intermediate predictions are intentionally excluded.",
            "CPU memory is sampled as process working set / RSS.",
            "GPU memory columns are populated only when using a CUDA device.",
        ],
    }

    common_filenames = runtime_bench.load_filename_filter(args.common_files_csv)

    keypoint_setup: dict[str, Any] | None = None
    if "keypoint" in methods:
        keypoint_setup = runtime_bench.benchmark_keypoint_setup(args, device)
        keypoint_filenames = list(getattr(keypoint_setup["dataset"], "images"))
        metadata["keypoint_dataset_size"] = len(keypoint_filenames)
        if common_filenames is None and "segmentation" in methods:
            common_filenames = set(keypoint_filenames)

    if "segmentation" in methods:
        seg_setup = runtime_bench.benchmark_segmentation_setup(args, device)
        seg_images = runtime_bench.list_segmentation_images(
            args.seg_input_dir,
            args.seg_file_glob,
            common_filenames,
        )
        seg_items = seg_images[: args.warmup + args.limit] if args.limit > 0 else seg_images
        metadata["segmentation_image_count"] = len(seg_items)
        rows.extend(
            benchmark_with_warmup(
                seg_items,
                args.warmup,
                lambda path: benchmark_segmentation_one(path, seg_setup, args),
            )
        )

    if "keypoint" in methods and keypoint_setup is not None:
        dataset = keypoint_setup["dataset"]
        kp_indices = list(range(len(dataset)))
        kp_items = kp_indices[: args.warmup + args.limit] if args.limit > 0 else kp_indices
        metadata["keypoint_benchmark_count"] = len(kp_items)
        rows.extend(
            benchmark_with_warmup(
                kp_items,
                args.warmup,
                lambda idx: benchmark_keypoint_one(idx, keypoint_setup, args),
            )
        )

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    per_image_df = pd.DataFrame(rows)
    summary_df = build_summary(per_image_df)

    per_image_path = args.output_dir / "per_image_memory.csv"
    summary_path = args.output_dir / "summary_by_method.csv"
    metadata_path = args.output_dir / "run_metadata.json"

    per_image_df.to_csv(per_image_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    runtime_bench.write_metadata(metadata_path, metadata)

    print(f"Saved per-image memory stats to: {per_image_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
