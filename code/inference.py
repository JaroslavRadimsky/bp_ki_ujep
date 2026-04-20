#!/usr/bin/env python3
"""
infer_atlas_sw_post.py

Atlas UNet inference na PNG obrázcích pomocí MONAI sliding_window_inference
+ stejný postprocessing jako v infer_atlas.py:

- prahování podle jistoty softmax (CONF_THRESH)
- čištění jednotlivých tříd (odstranění malých komponent, ponechání největší,
  opening/closing, fill holes)
- volitelné přerelabelování tříd 1..6 podle vertikální pozice (odshora dolů)
- uložení barevné masky + overlay

Použití (z rootu projektu, kde existuje balík Src.*):
    python -m Src.Atlas.infer_atlas_sw_post \
        --model_path Src/Models/Atlas-heqv-multi-100-fold-4-final.pth \
        --input_dir ./input_pngs \
        --output_dir ./output_masks_sw \
        --sw_batch_size 8 \
        --overlap 0.25 \
        --conf_thresh 0.5
"""

import os
import sys
import glob
import argparse

import numpy as np
from PIL import Image
import cv2

import torch
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, ScaleIntensityd, SpatialPadd, ToTensord

# --------------------------
# Import projektu (Src.*)
# --------------------------
# Pokud spouštíš jako `python -m Src.Atlas.infer_atlas_sw_post`, není třeba.
# Pokud spouštíš přímo soubor, pomůže to najít Src.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from Src.Utils.image_utils import HistogramEqualizationd  # stejný HEQ jako při tréninku

# --------------------------
# Postprocessing knihovny
# --------------------------
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes
from skimage.measure import label, regionprops


# ==========================
# 1) Konfigurace
# ==========================

NUM_CLASSES = 7       # 6 obratlů + background
IN_CHANNELS = 3       # RGB
ROI_SIZE = (512, 512) # okno pro sliding window
CONF_THRESH_DEFAULT = 0.5

def bgr_to_rgb(c):
    b, g, r = c
    return (r, g, b)

# paleta jako v testing_visu.py (OpenCV = BGR)
VIVID_COLORS_MULTI = {
    1: (  0, 255, 255),
    2: (255,   0, 255),
    3: (255, 255,   0),
    4: (  0, 128, 255),
    5: (255,   0,   0),
    6: (  0,   0, 255),
}

CLASS_COLORS = {
    0: (0, 0, 0),  # background
    1: bgr_to_rgb(VIVID_COLORS_MULTI[1]),
    2: bgr_to_rgb(VIVID_COLORS_MULTI[2]),
    3: bgr_to_rgb(VIVID_COLORS_MULTI[3]),
    4: bgr_to_rgb(VIVID_COLORS_MULTI[4]),
    5: bgr_to_rgb(VIVID_COLORS_MULTI[5]),
    6: bgr_to_rgb(VIVID_COLORS_MULTI[6]),
}

def mask_to_color_bgr(mask: np.ndarray) -> np.ndarray:
    """mask (H,W) -> BGR image (H,W,3) podle VIVID_COLORS_MULTI"""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, NUM_CLASSES):  # 1..6
        colored[mask == c] = VIVID_COLORS_MULTI[c]
    return colored


def blend_like_testing_visu(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5):
    """
    Vrací (image_bgr, colored_mask_bgr, blended_hat_bgr) stejně jako testing_visu (multi).
    image_bgr: uint8 BGR
    mask: uint8 [H,W] s třídami 0..6
    """
    colored_mask = mask_to_color_bgr(mask)

    overlay = image_bgr.copy()
    overlay[mask > 0] = colored_mask[mask > 0]

    blended = cv2.addWeighted(overlay, alpha, image_bgr, 1 - alpha, 0)
    return image_bgr, colored_mask, blended


# ==========================
# 2) Model + transformace
# ==========================

def build_model(num_classes: int = NUM_CLASSES, in_channels: int = IN_CHANNELS) -> torch.nn.Module:
    """Postaví stejný UNet jako v tréninku."""
    return UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )


def load_trained_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Načte uložený .pth model (zkusí několik typických formátů checkpointu)."""
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
        raise ValueError("Neznámý formát checkpointu (čekal jsem dict).")

    print("Checkpoint keys sample:", list(state_dict.keys())[:5])
    print("Num keys:", len(state_dict))

    
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    w = next(iter(model.parameters())).detach().float().cpu()
    print(f"First param stats after load: min={w.min().item():.6f} mean={w.mean().item():.6f} max={w.max().item():.6f} std={w.std().item():.6f}")
    
    return model


def get_inference_transform_sw(roi_size=ROI_SIZE):
    """
    Transformace pro sliding-window inference:
      - HEQ + normalizace
      - SpatialPad: zajistí min. velikost roi_size (když je snímek menší)
      - ToTensor
    Neprovádíme resize/crop na 512x512, chceme zachovat původní rozlišení.
    """
    return Compose([
        HistogramEqualizationd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        SpatialPadd(keys=["image", "image_raw"], spatial_size=roi_size),
        ToTensord(keys=["image", "image_raw"]),
    ])


# ==========================
# 3) Postprocessing
# ==========================

def clean_class(mask_cls: np.ndarray, min_size: int = 500) -> np.ndarray:
    """
    Vyčistí masku jedné třídy:
      - odstraní malé komponenty,
      - nechá jen největší,
      - provede opening/closing,
      - zalepí díry.
    mask_cls: bool pole (H, W)
    """
    if not mask_cls.any():
        return mask_cls

    labeled = label(mask_cls)  # 0 = background, 1..N = komponenty
    regions = regionprops(labeled)

    # 1) vyhození malých komponent
    for r in regions:
        if r.area < min_size:
            labeled[labeled == r.label] = 0

    labeled = label(labeled > 0)
    regions = regionprops(labeled)
    if len(regions) == 0:
        return np.zeros_like(mask_cls, dtype=bool)

    # 2) necháme největší komponentu
    areas = [r.area for r in regions]
    largest_label = regions[int(np.argmax(areas))].label
    mask_main = (labeled == largest_label)

    # 3) morfologické vyhlazení
    struct = np.ones((3, 3), dtype=bool)
    mask_main = binary_opening(mask_main, structure=struct)
    mask_main = binary_closing(mask_main, structure=struct)

    # 4) zalepit díry
    mask_main = binary_fill_holes(mask_main)

    return mask_main


def postprocess_mask(pred_mask: np.ndarray,
                     num_classes: int = NUM_CLASSES,
                     min_size: int = 500) -> np.ndarray:
    """
    Postprocessing celé masky:
      - pro každou třídu (1..num_classes-1) se aplikuje clean_class.
    pred_mask: (H, W), hodnoty 0..num_classes-1
    """
    H, W = pred_mask.shape
    cleaned = np.zeros((H, W), dtype=np.uint8)

    for cls in range(1, num_classes):  # 1..6, 0 = background
        cls_mask = (pred_mask == cls)
        cls_clean = clean_class(cls_mask, min_size=min_size)
        cleaned[cls_clean] = cls

    return cleaned


def relabel_by_vertical_position(mask: np.ndarray,
                                 class_ids=(1, 2, 3, 4, 5, 6)) -> np.ndarray:
    """
    Přerelabeluje třídy 1..6 podle vertikální pozice (odshora dolů).
    Užitočné, pokud model někdy „prohodí“ čísla obratlů.
    """
    centroids = []

    for cls in class_ids:
        m = (mask == cls)
        if not m.any():
            continue
        lab = label(m)
        regs = regionprops(lab)
        if not regs:
            continue
        # vezmeme největší komponentu
        areas = [r.area for r in regs]
        r_main = regs[int(np.argmax(areas))]
        y, _x = r_main.centroid
        centroids.append((cls, y))

    centroids_sorted = sorted(centroids, key=lambda t: t[1])

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    for new_label, (old_label, _) in enumerate(centroids_sorted, start=1):
        new_mask[mask == old_label] = new_label

    return new_mask


# ==========================
# 4) Vizualizace
# ==========================

def mask_to_color(mask: np.ndarray) -> Image.Image:
    """mask (H,W) -> RGB obrázek podle CLASS_COLORS"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, rgb in CLASS_COLORS.items():
        color_img[mask == cls] = rgb
    return Image.fromarray(color_img, mode="RGB")


# ==========================
# 5) Inference pro jeden obrázek (SW + post)
# ==========================

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
    """
    Vrací:
      - final_mask: (H_orig, W_orig) s třídami 0..6
      - overlay_base: RGB obraz (H_orig, W_orig)
    """
    # 1) načtení původního PNG
    img_pil = Image.open(img_path).convert("RGB")
    img_np_hw3 = np.array(img_pil, dtype=np.float32)        # (H, W, 3)
    orig_h, orig_w = img_np_hw3.shape[:2]

    # (H, W, 3) -> (3, H, W)
    img_np = np.transpose(img_np_hw3, (2, 0, 1))            # (3, H, W)

    data = {"image": img_np, "image_raw": img_np.copy()}
    data = transform(data)

    img_tensor = data["image"].unsqueeze(0).to(device)      # (1, 3, Hpad, Wpad)
    raw_tensor = data["image_raw"]                            # (3, Hpad, Wpad)
    
    it = img_tensor.detach().float().cpu()
    # print(f"  input shape: {tuple(it.shape)}")
    # print(f"  input stats: min={it.min().item():.4f} mean={it.mean().item():.4f} max={it.max().item():.4f} std={it.std().item():.4f}")


    # 2) sliding window inference -> logits (1, 7, Hpad, Wpad)
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

    lg = logits.detach().float().cpu()
    # print(f"  logits stats: min={lg.min().item():.4f} mean={lg.mean().item():.4f} max={lg.max().item():.4f} std={lg.std().item():.4f}")
    # print(f"  logits per-class mean: {lg.mean(dim=(0,2,3)).numpy()}")


    probs = torch.softmax(logits, dim=1)                     # (1, 7, Hpad, Wpad)
    max_prob, pred = probs.max(dim=1)                        # (1, Hpad, Wpad)

    pred_mask = pred.squeeze(0).cpu().numpy()                # (Hpad, Wpad)
    max_prob_np = max_prob.squeeze(0).cpu().numpy()          # (Hpad, Wpad)

    # DIAGNOSTIKA
    unique_before = np.unique(pred_mask)
    # print(f"  unique pred before thresh: {unique_before[:20]} (n={len(unique_before)})")
    # print(f"  max_prob stats: min={max_prob_np.min():.3f} mean={max_prob_np.mean():.3f} max={max_prob_np.max():.3f}")
    # print(f"  pixels >= conf_thresh({conf_thresh}): {(max_prob_np >= conf_thresh).mean()*100:.2f}%")


    # 3) vynulování nejistých pixelů
    pred_mask[max_prob_np < conf_thresh] = 0
    
    unique_after = np.unique(pred_mask)
    # print(f"  unique pred after thresh: {unique_after[:20]} (n={len(unique_after)})")


    # 4) postprocessing (stejný jako dřív)
    cleaned_mask = postprocess_mask(pred_mask, num_classes=NUM_CLASSES, min_size=min_size)
    if apply_relabel:
        cleaned_mask = relabel_by_vertical_position(cleaned_mask)

    # 5) oříznutí zpět na původní velikost (kvůli paddingu, když orig < ROI)
    cleaned_mask = cleaned_mask[:orig_h, :orig_w].astype(np.uint8)
    
    unique_clean = np.unique(cleaned_mask)
    # print(f"  unique after postprocess: {unique_clean[:20]} (n={len(unique_clean)})")


    raw_img_np = raw_tensor.permute(1, 2, 0).cpu().numpy()   # (Hpad, Wpad, 3)
    raw_img_np = np.clip(raw_img_np, 0, 255).astype(np.uint8)
    raw_img_np = raw_img_np[:orig_h, :orig_w, :]
    overlay_base = Image.fromarray(raw_img_np, mode="RGB")

    return cleaned_mask, overlay_base


# ==========================
# 6) CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Atlas UNet: sliding window inference + postprocessing na PNG souborech."
    )
    parser.add_argument("--model_path", type=str, required=True, help="Cesta k uloženému .pth modelu.")
    parser.add_argument("--input_dir", type=str, required=True, help="Složka s vstupními PNG obrázky.")
    parser.add_argument("--output_dir", type=str, required=True, help="Složka pro uložení výstupů.")

    parser.add_argument("--no_relabel", action="store_true", help="Nevykonávat anatomické přerelabelování 1..6.")
    parser.add_argument("--conf_thresh", type=float, default=CONF_THRESH_DEFAULT,
                        help=f"Prahová hodnota jistoty softmax (default {CONF_THRESH_DEFAULT}).")
    parser.add_argument("--min_size", type=int, default=500, help="Min. velikost komponenty v pixelech (default 500).")

    parser.add_argument("--sw_batch_size", type=int, default=4, help="Kolik oken najednou na GPU (default 4).")
    parser.add_argument("--overlap", type=float, default=0.25, help="Overlap oken (default 0.25).")
    parser.add_argument("--limit", type=int, default=0, help="0 = bez limitu, jinak max počet obrázků.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Použité zařízení: {device}")
    print(f"Načítám model z: {args.model_path}")

    model = load_trained_model(args.model_path, device)
    transform = get_inference_transform_sw(roi_size=ROI_SIZE)

    image_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.png")))
    if not image_paths:
        print("Ve složce input_dir jsem nenašel žádné PNG soubory.")
        return

    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Nalezeno {len(image_paths)} obrázků.")
    print(f"ROI: {ROI_SIZE}, overlap={args.overlap}, sw_batch_size={args.sw_batch_size}")
    print(f"conf_thresh={args.conf_thresh}, min_size={args.min_size}, relabel={not args.no_relabel}")

    for idx, img_path in enumerate(image_paths, start=1):
        fname = os.path.basename(img_path)
        
        # if idx >= 10:
        #     print(f"Test run stop.")
        #     break
        
        print(f"[{idx}/{len(image_paths)}] Zpracovávám: {fname}")
    

        final_mask, overlay_base = run_inference_on_image_sw(
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

                # --- vizualizace jako testing_visu.py ---
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
        if img_bgr is None:
            print(f"  ! Nelze načíst obrázek přes cv2: {img_path}")
            continue

        # jistota: velikost musí sedět
        img_bgr = img_bgr[: final_mask.shape[0], : final_mask.shape[1], :]

        image_bgr, maskhat_bgr, blended_hat_bgr = blend_like_testing_visu(
            img_bgr, final_mask, alpha=0.5
        )

        stem = os.path.splitext(fname)[0]
        out_image   = os.path.join(args.output_dir, f"{stem}_image.png")
        out_maskhat = os.path.join(args.output_dir, f"{stem}_maskhat.png")
        out_blended = os.path.join(args.output_dir, f"{stem}_blended_hat.png")

        cv2.imwrite(out_image, image_bgr)
        cv2.imwrite(out_maskhat, maskhat_bgr)
        cv2.imwrite(out_blended, blended_hat_bgr)

        print(f"  → uloženo: {out_image}")
        print(f"  → uloženo: {out_maskhat}")
        print(f"  → uloženo: {out_blended}")


    print("Hotovo.")


if __name__ == "__main__":
    main()
