#!/usr/bin/env python3
"""Run atlas vertebra segmentation inference with sliding-window prediction.

The script loads a trained MONAI UNet checkpoint, applies the same image
preprocessing used during training, performs sliding-window inference, runs a
postprocessing pipeline on the predicted mask, and stores three visualization
outputs per image:
1. the original image,
2. the colorized predicted mask,
3. the blended overlay.

Example:
    python code/inference.py ^
        --model_path .\\Models\\Atlas-heqv-multi-100-fold-4-final.pth ^
        --input_dir .\\input_pngs ^
        --output_dir .\\output_masks_sw ^
        --sw_batch_size 8 ^
        --overlap 0.25 ^
        --conf_thresh 0.5
"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd, ToTensord
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from skimage.measure import label, regionprops

from image_utils import HistogramEqualizationd


# --- Inference configuration ------------------------------------------------

NUM_CLASSES = 7
IN_CHANNELS = 3
ROI_SIZE = (512, 512)
CONF_THRESH_DEFAULT = 0.5


def bgr_to_rgb(color_bgr):
    """Convert one OpenCV-style BGR color tuple to RGB."""
    b, g, r = color_bgr
    return (r, g, b)


# Use the same vivid palette as the original visualization utilities.
VIVID_COLORS_MULTI = {
    1: (0, 255, 255),
    2: (255, 0, 255),
    3: (255, 255, 0),
    4: (0, 128, 255),
    5: (255, 0, 0),
    6: (0, 0, 255),
}

CLASS_COLORS = {
    0: (0, 0, 0),
    1: bgr_to_rgb(VIVID_COLORS_MULTI[1]),
    2: bgr_to_rgb(VIVID_COLORS_MULTI[2]),
    3: bgr_to_rgb(VIVID_COLORS_MULTI[3]),
    4: bgr_to_rgb(VIVID_COLORS_MULTI[4]),
    5: bgr_to_rgb(VIVID_COLORS_MULTI[5]),
    6: bgr_to_rgb(VIVID_COLORS_MULTI[6]),
}


def mask_to_color_bgr(mask: np.ndarray) -> np.ndarray:
    """Convert a class mask to a BGR visualization image."""
    height, width = mask.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id in range(1, NUM_CLASSES):
        colored[mask == class_id] = VIVID_COLORS_MULTI[class_id]
    return colored


def blend_like_testing_visu(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """Create the same image, mask, and overlay trio as the original visualizer."""
    colored_mask = mask_to_color_bgr(mask)

    overlay = image_bgr.copy()
    overlay[mask > 0] = colored_mask[mask > 0]

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return image_bgr, colored_mask, blended


# --- Model construction and preprocessing ----------------------------------

def build_model(num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS) -> torch.nn.Module:
    """Build the UNet architecture used during atlas training."""
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def load_trained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained checkpoint and return an evaluation-ready model."""
    model = build_model().to(device)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format: expected a dictionary.")

    print("Checkpoint keys sample:", list(state_dict.keys())[:5])
    print("Number of checkpoint keys:", len(state_dict))

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    weights = next(iter(model.parameters())).detach().float().cpu()
    print(
        "First parameter stats after load: "
        f"min={weights.min().item():.6f} "
        f"mean={weights.mean().item():.6f} "
        f"max={weights.max().item():.6f} "
        f"std={weights.std().item():.6f}"
    )

    return model


def get_inference_transform_sw(roi_size=ROI_SIZE):
    """Create the preprocessing pipeline used before sliding-window inference.

    The transform intentionally preserves the original image resolution. Images
    are only padded up to ``roi_size`` when they are smaller than the sliding
    window expected by the model.
    """
    return Compose(
        [
            HistogramEqualizationd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            SpatialPadd(keys=["image", "image_raw"], spatial_size=roi_size),
            ToTensord(keys=["image", "image_raw"]),
        ]
    )


# --- Postprocessing ---------------------------------------------------------

def clean_class(mask_cls: np.ndarray, min_size: int = 500) -> np.ndarray:
    """Clean one class-specific binary mask.

    The procedure removes tiny connected components, keeps only the dominant
    component, smooths the contour with simple morphology, and fills small
    interior holes.
    """
    if not mask_cls.any():
        return mask_cls

    labeled = label(mask_cls)
    regions = regionprops(labeled)

    # Remove tiny components that are very unlikely to represent vertebrae.
    for region in regions:
        if region.area < min_size:
            labeled[labeled == region.label] = 0

    labeled = label(labeled > 0)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return np.zeros_like(mask_cls, dtype=bool)

    # Keep only the largest surviving component for the current class.
    areas = [region.area for region in regions]
    largest_label = regions[int(np.argmax(areas))].label
    mask_main = labeled == largest_label

    # Apply light morphology to stabilize the final contour.
    structure = np.ones((3, 3), dtype=bool)
    mask_main = binary_opening(mask_main, structure=structure)
    mask_main = binary_closing(mask_main, structure=structure)
    mask_main = binary_fill_holes(mask_main)

    return mask_main


def postprocess_mask(pred_mask: np.ndarray, num_classes: int = NUM_CLASSES, min_size: int = 500) -> np.ndarray:
    """Run class-wise cleanup over the full predicted mask."""
    height, width = pred_mask.shape
    cleaned = np.zeros((height, width), dtype=np.uint8)

    for class_id in range(1, num_classes):
        cls_mask = pred_mask == class_id
        cls_clean = clean_class(cls_mask, min_size=min_size)
        cleaned[cls_clean] = class_id

    return cleaned


def relabel_by_vertical_position(mask: np.ndarray, class_ids=(1, 2, 3, 4, 5, 6)) -> np.ndarray:
    """Relabel vertebra classes from top to bottom based on vertical position.

    This is useful when the network predicts plausible vertebra shapes but swaps
    their numeric class identities.
    """
    centroids = []

    for class_id in class_ids:
        class_mask = mask == class_id
        if not class_mask.any():
            continue

        labeled = label(class_mask)
        regions = regionprops(labeled)
        if not regions:
            continue

        # Use the dominant connected component to estimate vertical ordering.
        areas = [region.area for region in regions]
        main_region = regions[int(np.argmax(areas))]
        y_coord, _ = main_region.centroid
        centroids.append((class_id, y_coord))

    centroids_sorted = sorted(centroids, key=lambda item: item[1])

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for new_label, (old_label, _) in enumerate(centroids_sorted, start=1):
        new_mask[mask == old_label] = new_label

    return new_mask


# --- Visualization helpers --------------------------------------------------

def mask_to_color(mask: np.ndarray) -> Image.Image:
    """Convert a class mask to an RGB PIL image using the configured palette."""
    height, width = mask.shape
    color_img = np.zeros((height, width, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS.items():
        color_img[mask == class_id] = rgb
    return Image.fromarray(color_img, mode="RGB")


# --- Inference over one image -----------------------------------------------

@torch.no_grad()
def run_inference_on_image_sw(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    transform,
    roi_size=ROI_SIZE,
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    conf_thresh: float = CONF_THRESH_DEFAULT,
    apply_relabel: bool = True,
    min_size: int = 500,
) -> tuple[np.ndarray, Image.Image]:
    """Run the full segmentation pipeline for one image.

    Returns:
        A tuple ``(final_mask, overlay_base)`` where ``final_mask`` is the
        cleaned class mask cropped back to the original image size and
        ``overlay_base`` is the RGB base image used for visualization.
    """
    # Load the original RGB image and keep its native spatial resolution.
    img_pil = Image.open(img_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)
    orig_h, orig_w = img_np_hw3.shape[:2]

    # Convert the image to channel-first format expected by MONAI transforms.
    img_np = np.transpose(img_np_hw3, (2, 0, 1))
    data = {"image": img_np, "image_raw": img_np.copy()}
    data = transform(data)

    img_tensor = data["image"].unsqueeze(0).to(device)
    raw_tensor = data["image_raw"]

    # Run sliding-window inference to obtain dense per-class logits.
    logits = sliding_window_inference(
        inputs=img_tensor,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=model,
        overlap=overlap,
        mode="gaussian",
        device=device,
        sw_device=device,
    )

    # Convert logits to the winning class map and remove low-confidence pixels.
    probs = torch.softmax(logits, dim=1)
    max_prob, pred = probs.max(dim=1)

    pred_mask = pred.squeeze(0).cpu().numpy()
    max_prob_np = max_prob.squeeze(0).cpu().numpy()
    pred_mask[max_prob_np < conf_thresh] = 0

    # Apply geometric cleanup and optional anatomical class reordering.
    cleaned_mask = postprocess_mask(pred_mask, num_classes=NUM_CLASSES, min_size=min_size)
    if apply_relabel:
        cleaned_mask = relabel_by_vertical_position(cleaned_mask)

    # Crop away padding introduced before inference.
    cleaned_mask = cleaned_mask[:orig_h, :orig_w].astype(np.uint8)

    # Rebuild the padded raw image and crop it back to the original size.
    raw_img_np = raw_tensor.permute(1, 2, 0).cpu().numpy()
    raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)
    raw_img_np = raw_img_np[:orig_h, :orig_w, :]
    overlay_base = Image.fromarray(raw_img_np, mode="RGB")

    return cleaned_mask, overlay_base


# --- Command-line interface -------------------------------------------------

def main():
    """Parse CLI arguments and run inference over all PNG files in a folder."""
    parser = argparse.ArgumentParser(
        description="Atlas UNet sliding-window inference with postprocessing on PNG images."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth checkpoint.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input PNG images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where output images will be written.")

    parser.add_argument(
        "--no_relabel",
        action="store_true",
        help="Disable anatomical relabeling of vertebra classes after postprocessing.",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=CONF_THRESH_DEFAULT,
        help=f"Softmax confidence threshold applied before postprocessing (default {CONF_THRESH_DEFAULT}).",
    )
    parser.add_argument("--min_size", type=int, default=500, help="Minimum connected component size in pixels.")
    parser.add_argument("--sw_batch_size", type=int, default=4, help="Number of sliding windows processed at once.")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap between neighboring sliding windows.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of images to process. Use 0 for no limit.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Loading model from: {args.model_path}")

    model = load_trained_model(args.model_path, device)
    transform = get_inference_transform_sw(roi_size=ROI_SIZE)

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_paths:
        print("No PNG images were found in the input directory.")
        return

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Found {len(image_paths)} images.")
    print(f"ROI={ROI_SIZE}, overlap={args.overlap}, sw_batch_size={args.sw_batch_size}")
    print(f"conf_thresh={args.conf_thresh}, min_size={args.min_size}, relabel={not args.no_relabel}")

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        print(f"[{idx}/{len(image_paths)}] Processing: {filename}")

        final_mask, _overlay_base = run_inference_on_image_sw(
            model=model,
            img_path=img_path,
            device=device,
            transform=transform,
            roi_size=ROI_SIZE,
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap,
            conf_thresh=args.conf_thresh,
            apply_relabel=not args.no_relabel,
            min_size=args.min_size,
        )

        # Build the final visualization trio from the original OpenCV image.
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"  Failed to read image with OpenCV: {img_path}")
            continue

        img_bgr = img_bgr[: final_mask.shape[0], : final_mask.shape[1], :]
        image_bgr, maskhat_bgr, blended_hat_bgr = blend_like_testing_visu(img_bgr, final_mask, alpha=0.5)

        stem = os.path.splitext(filename)[0]
        out_image = os.path.join(args.output_dir, f"{stem}_image.png")
        out_maskhat = os.path.join(args.output_dir, f"{stem}_maskhat.png")
        out_blended = os.path.join(args.output_dir, f"{stem}_blended_hat.png")

        cv2.imwrite(out_image, image_bgr)
        cv2.imwrite(out_maskhat, maskhat_bgr)
        cv2.imwrite(out_blended, blended_hat_bgr)

        print(f"  Saved: {out_image}")
        print(f"  Saved: {out_maskhat}")
        print(f"  Saved: {out_blended}")

    print("Done.")


if __name__ == "__main__":
    main()
