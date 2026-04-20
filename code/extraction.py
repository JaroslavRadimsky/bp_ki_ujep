import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def unique_colors_bgr(img_bgr: np.ndarray) -> list[tuple[int, int, int]]:
    """Return unique BGR colors excluding black background (0,0,0)."""
    flat = img_bgr.reshape(-1, 3)
    colors = np.unique(flat, axis=0)
    colors = [tuple(map(int, c)) for c in colors]
    colors = [c for c in colors if c != (0, 0, 0)]
    return colors


def mask_from_color(img_bgr: np.ndarray, color_bgr: tuple[int, int, int], tol: int = 0) -> np.ndarray:
    """Binary mask of pixels matching a BGR color, optionally with tolerance."""
    b, g, r = color_bgr
    if tol <= 0:
        m = (img_bgr[:, :, 0] == b) & (img_bgr[:, :, 1] == g) & (img_bgr[:, :, 2] == r)
        return m.astype(np.uint8) * 255
    lower = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8)
    upper = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8)
    return cv2.inRange(img_bgr, lower, upper)

# Binary cleaning already happens in postprocessing part of inference, so this is not needed here.
# def clean_binary(mask_u8: np.ndarray, k: int = 5) -> np.ndarray:
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
#     m = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
#     m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=2)
#     return m


def main_contour(mask_u8: np.ndarray):
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def approx_poly_n(contour: np.ndarray, n_vertices: int) -> np.ndarray | None:
    """
    Try to approximate contour to exactly n_vertices.
    First tries cv2.approxPolyN (if available), then falls back to eps-sweep approxPolyDP.
    Returns Nx2 array or None.
    """
    try:
        approx = cv2.approxPolyN(contour, n_vertices, True)
        if approx is not None and len(approx) == n_vertices:
            return approx.reshape(-1, 2)
    except Exception:
        pass

    peri = cv2.arcLength(contour, True)
    for eps in np.linspace(0.005, 0.20, 40):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == n_vertices:
            return approx.reshape(-1, 2)

    return None


def pairwise_min_distance(pts: np.ndarray) -> float:
    """Return minimum pairwise Euclidean distance between points."""
    pts = pts.astype(np.float32)
    min_dist = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            min_dist = min(min_dist, d)
    return 0.0 if not np.isfinite(min_dist) else min_dist


def polygon_min_edge_length(pts: np.ndarray) -> float:
    """Return minimum edge length of a closed polygon."""
    pts = pts.astype(np.float32)
    min_edge = np.inf
    for i in range(len(pts)):
        p1 = pts[i]
        p2 = pts[(i + 1) % len(pts)]
        d = float(np.linalg.norm(p1 - p2))
        min_edge = min(min_edge, d)
    return 0.0 if not np.isfinite(min_edge) else min_edge


def polygon_area_abs(pts: np.ndarray) -> float:
    """Return absolute polygon area."""
    pts = pts.astype(np.float32).reshape(-1, 1, 2)
    return float(abs(cv2.contourArea(pts)))


def merge_close_points(pts: np.ndarray, merge_dist: float = 6.0) -> np.ndarray:
    """
    Merge points that are closer than merge_dist by replacing them with their mean.
    Useful when approximation returns nearly duplicated vertices.
    """
    if pts is None or len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts_list = [p.astype(np.float32) for p in pts]
    merged = []

    while pts_list:
        ref = pts_list.pop(0)
        group = [ref]
        rest = []

        for p in pts_list:
            if np.linalg.norm(p - ref) < merge_dist:
                group.append(p)
            else:
                rest.append(p)

        merged.append(np.mean(group, axis=0))
        pts_list = rest

    return np.array(merged, dtype=np.float32)


def quad_is_valid(
    pts: np.ndarray,
    min_pair_dist: float = 8.0,
    min_area: float = 50.0,
    min_edge: float = 6.0,
) -> bool:
    """
    Validate 4-point polygon to reject degenerate quads
    (duplicated/very close vertices, tiny area, tiny edges).
    """
    if pts is None or len(pts) != 4:
        return False

    pts = pts.astype(np.float32)

    if pairwise_min_distance(pts) < min_pair_dist:
        return False

    if polygon_area_abs(pts) < min_area:
        return False

    if polygon_min_edge_length(pts) < min_edge:
        return False

    return True


def approx_quad(contour: np.ndarray) -> np.ndarray | None:
    """
    Robust 4-corner extraction.
    1) try polygon approximation
    2) merge almost duplicated points if needed
    3) fallback to minAreaRect on convex hull
    """
    quad = approx_poly_n(contour, 4)
    if quad is not None:
        quad = quad.astype(np.float32)
        if quad_is_valid(quad):
            return quad

        merged = merge_close_points(quad, merge_dist=6.0)
        if len(merged) == 4 and quad_is_valid(merged):
            return merged.astype(np.float32)

    hull = cv2.convexHull(contour)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect).astype(np.float32)

    if quad_is_valid(box, min_pair_dist=5.0, min_area=30.0, min_edge=5.0):
        return box

    return None


def order_quad_clockwise(pts_xy: np.ndarray) -> np.ndarray:
    """
    Basic fallback ordering of quad corners as:
    TL, TR, BR, BL

    Uses a simple top/bottom split by y and left/right split by x.
    This is used only as a fallback when no previous vertebra reference exists.
    """
    pts = pts_xy.astype(np.float32)

    y_order = np.argsort(pts[:, 1])
    top = pts[y_order[:2]]
    bottom = pts[y_order[2:]]

    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]

    tl, tr = top
    bl, br = bottom

    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_quad_from_previous_bottom(
    pts_xy: np.ndarray,
    prev_bl: np.ndarray | None,
    prev_br: np.ndarray | None,
) -> np.ndarray:
    """
    Order current vertebra corners as TL, TR, BR, BL using previous vertebra bottom points.

    Strategy:
    - choose the pair of current points that best matches previous BL/BR
      -> this pair becomes the TOP pair of the current vertebra
    - remaining two points become the BOTTOM pair
    - within the top pair:
        point closer to prev_bl => TL
        point closer to prev_br => TR
    - within the bottom pair:
        left/right determined by x coordinate
    """
    pts = pts_xy.astype(np.float32)

    if prev_bl is None or prev_br is None or len(pts) != 4:
        return order_quad_clockwise(pts)

    best_score = np.inf
    best_top_idx = None
    best_assignment = None

    for i in range(4):
        for j in range(i + 1, 4):
            pair = pts[[i, j]]

            s1 = np.linalg.norm(pair[0] - prev_bl) + np.linalg.norm(pair[1] - prev_br)
            s2 = np.linalg.norm(pair[0] - prev_br) + np.linalg.norm(pair[1] - prev_bl)

            if s1 <= s2:
                score = s1
                assignment = (0, 1)  # pair[0] -> TL, pair[1] -> TR
            else:
                score = s2
                assignment = (1, 0)  # pair[1] -> TL, pair[0] -> TR

            if score < best_score:
                best_score = score
                best_top_idx = (i, j)
                best_assignment = assignment

    if best_top_idx is None:
        return order_quad_clockwise(pts)

    top_idx = list(best_top_idx)
    bottom_idx = [k for k in range(4) if k not in top_idx]

    top_pair = pts[top_idx]
    bottom_pair = pts[bottom_idx]

    tl = top_pair[best_assignment[0]]
    tr = top_pair[best_assignment[1]]

    bottom_pair = bottom_pair[np.argsort(bottom_pair[:, 0])]
    bl, br = bottom_pair[0], bottom_pair[1]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def order_triangle_bl_br_apex(pts_xy: np.ndarray) -> np.ndarray:
    """
    Order 3 points to (bottom-left, bottom-right, apex).
    Apex = smallest y. Remaining two ordered by x (left/right).
    """
    pts = pts_xy.astype(np.float32)

    apex_idx = np.argmin(pts[:, 1])
    apex = pts[apex_idx]

    base = np.delete(pts, apex_idx, axis=0)
    if base[0, 0] <= base[1, 0]:
        bl, br = base[0], base[1]
    else:
        bl, br = base[1], base[0]

    return np.stack([bl, br, apex], axis=0)


def refine_subpix(mask_u8: np.ndarray, corners_xy: np.ndarray, win: int = 7) -> np.ndarray:
    """Subpixel refinement using cornerSubPix on a blurred grayscale version of the mask."""
    gray = cv2.GaussianBlur(mask_u8, (5, 5), 0)
    c = corners_xy.astype(np.float32).reshape(-1, 1, 2)
    term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-4)
    refined = cv2.cornerSubPix(gray, c, (win, win), (-1, -1), term)
    return refined.reshape(-1, 2)


def centroid_of_mask(mask_u8: np.ndarray) -> tuple[float, float] | None:
    m = cv2.moments(mask_u8, binaryImage=True)
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return float(cx), float(cy)


def bottom_edge_endpoints_from_contour(contour: np.ndarray, band_frac: float = 0.12, min_band_px: int = 8):
    """Robust endpoints of bottom edge even if skewed."""
    pts = contour.reshape(-1, 2).astype(np.float32)

    y = pts[:, 1]
    min_y = float(np.min(y))
    max_y = float(np.max(y))
    height = max_y - min_y

    band = max(min_band_px, int(round(band_frac * height)))
    thresh = max_y - band

    bottom_pts = pts[y >= thresh]
    if len(bottom_pts) < 2:
        idx = np.argsort(y)[-2:]
        bottom_pts = pts[idx]

    bl = bottom_pts[np.argmin(bottom_pts[:, 0])]
    br = bottom_pts[np.argmax(bottom_pts[:, 0])]

    if bl[0] > br[0]:
        bl, br = br, bl

    return bl, br


def apex_top_point_from_contour(contour: np.ndarray, y_tol: int = 2) -> np.ndarray:
    """Return apex (topmost point) using a small y-band for stability."""
    pts = contour.reshape(-1, 2).astype(np.float32)
    min_y = float(np.min(pts[:, 1]))
    band = pts[pts[:, 1] <= (min_y + y_tol)]
    if len(band) == 0:
        band = pts[np.argmin(pts[:, 1]): np.argmin(pts[:, 1]) + 1]
    apex = np.mean(band, axis=0)
    return apex


def labelme_point(label: str, x: float, y: float) -> dict:
    return {
        "label": label,
        "points": [[float(x), float(y)]],
        "group_id": None,
        "description": "",
        "shape_type": "point",
        "flags": {},
    }


def build_labelme_json(image_path: str, w: int, h: int, shapes: list[dict], version: str = "5.2.1") -> dict:
    return {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": Path(image_path).name,
        "imageData": "",
        "imageHeight": int(h),
        "imageWidth": int(w),
    }


def parse_labels(labels_str: str | None) -> list[str] | None:
    if labels_str is None:
        return None
    labels = [x.strip() for x in labels_str.split(",") if x.strip()]
    return labels if labels else None


def process_one_mask(mask_path: Path, out_json_path: Path, labels: list[str] | None, tol: int, min_area: int) -> tuple[bool, str]:
    """
    Process single mask PNG -> LabelMe JSON.
    Returns (success, message). No prints here.
    """
    img = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, f"Cannot read mask: {mask_path}"

    h, w = img.shape[:2]
    colors = unique_colors_bgr(img)
    if not colors:
        return False, f"No non-black colors found: {mask_path}"

    vertebra_items = []
    for c in colors:
        m = mask_from_color(img, c, tol=tol)
        # m = clean_binary(m, k=5)

        if cv2.countNonZero(m) < min_area:
            continue

        ctr = centroid_of_mask(m)
        if ctr is None:
            continue

        vertebra_items.append({"color": c, "mask": m, "centroid": ctr})

    if not vertebra_items:
        return False, f"No usable regions after filtering: {mask_path}"

    vertebra_items.sort(key=lambda it: it["centroid"][1])
    n = len(vertebra_items)

    requested_labels = labels

    if requested_labels is None:
        default = ["C2", "C3", "C4", "C5", "C6", "C7"]
        if n == len(default):
            names = default
        else:
            names = [f"V{i+1}" for i in range(n)]
    else:
        if len(requested_labels) == n:
            names = requested_labels
        elif len(requested_labels) < n:
            start = n - len(requested_labels)
            vertebra_items = vertebra_items[start:]
            names = requested_labels
            n = len(vertebra_items)
        else:
            names = requested_labels[:n]

    shapes: list[dict] = []

    prev_bl = None
    prev_br = None

    for it, name in zip(vertebra_items, names):
        m = it["mask"]
        cnt = main_contour(m)
        if cnt is None or cv2.contourArea(cnt) < min_area:
            continue

        if name == "C2":
            tri = approx_poly_n(cnt, 3)

            if tri is not None:
                tri = order_triangle_bl_br_apex(tri)
                tri = refine_subpix(m, tri, win=7)
                tri = order_triangle_bl_br_apex(tri)

                bl, br, apex_ref = tri
            else:
                bl, br = bottom_edge_endpoints_from_contour(cnt, band_frac=0.5, min_band_px=8)
                apex = apex_top_point_from_contour(cnt, y_tol=3)
                pts3 = np.stack([bl, br, apex], axis=0)
                pts3 = refine_subpix(m, pts3, win=7)
                bl, br, apex_ref = pts3[0], pts3[1], pts3[2]

            shapes.append(labelme_point("C2 bottom left", bl[0], bl[1]))
            shapes.append(labelme_point("C2 bottom right", br[0], br[1]))
            shapes.append(labelme_point("C2 centroid", apex_ref[0], apex_ref[1]))

            prev_bl = bl.astype(np.float32)
            prev_br = br.astype(np.float32)
            continue

        quad = approx_quad(cnt)
        if quad is None:
            continue

        quad = refine_subpix(m, quad, win=7)

        if not quad_is_valid(quad, min_pair_dist=5.0, min_area=30.0, min_edge=4.0):
            rect = cv2.minAreaRect(cv2.convexHull(cnt))
            quad = cv2.boxPoints(rect).astype(np.float32)

        quad = order_quad_from_previous_bottom(quad, prev_bl, prev_br)
        tl, tr, br, bl = quad

        shapes.append(labelme_point(f"{name} top left", tl[0], tl[1]))
        shapes.append(labelme_point(f"{name} top right", tr[0], tr[1]))
        shapes.append(labelme_point(f"{name} bottom right", br[0], br[1]))
        shapes.append(labelme_point(f"{name} bottom left", bl[0], bl[1]))

        prev_bl = bl.astype(np.float32)
        prev_br = br.astype(np.float32)

    data = build_labelme_json(mask_path.name, w, h, shapes, version="5.2.1")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    if len(shapes) == 0:
        return False, f"No points produced: {mask_path}"

    return True, f"OK: {mask_path.name} -> {out_json_path.name} ({len(shapes)} points)"


def iter_masks(input_path: Path, pattern: str, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        return []

    if recursive:
        return sorted(p for p in input_path.rglob(pattern) if p.is_file())
    return sorted(p for p in input_path.glob(pattern) if p.is_file())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input mask PNG file or directory with PNG masks.")
    ap.add_argument("--out_dir", required=True, help="Output directory for LabelMe JSON files.")
    ap.add_argument("--labels", default=None, help="Comma-separated labels top-to-bottom, e.g. 'C3,C4,C5,C6,C7'")
    ap.add_argument("--tol", type=int, default=0, help="Color tolerance (0 = exact). Use 1-5 if needed.")
    ap.add_argument("--min_area", type=int, default=300, help="Minimum area for a color region to be considered.")
    ap.add_argument("--recursive", action="store_true", help="Scan input directory recursively for *.png masks.")
    ap.add_argument("--verbose", action="store_true", help="Print per-file status + final summary.")
    ap.add_argument("--pattern", type=str, default="*_mask_color.png", help="Glob pattern pro výběr masek (default '*_mask_color.png').")
    args = ap.parse_args()

    input_path = Path(args.inp)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = parse_labels(args.labels)
    masks = iter_masks(
        input_path=input_path,
        pattern=args.pattern,
        recursive=args.recursive,
    )
    if not masks:
        raise FileNotFoundError(f"No PNG masks found in: {input_path}")

    ok = 0
    fail = 0

    for mp in masks:
        print(f"Processing: {mp}")
        out_json = out_dir / (mp.stem + ".json")
        success, msg = process_one_mask(
            mask_path=mp,
            out_json_path=out_json,
            labels=labels,
            tol=args.tol,
            min_area=args.min_area,
        )
        if success:
            ok += 1
        else:
            fail += 1
        if args.verbose:
            print(msg)

    if args.verbose:
        print(f"Done. OK={ok}  FAIL={fail}  Total={len(masks)}")


if __name__ == "__main__":
    main()