#!/usr/bin/env python3
"""Build static drone-leg occlusion masks for the birdseye payload

The drone leg is bolted to the airframe, so on cameras 2 and 4 it occupies a fixed pixel
region in every frame. A single mask per camera therefore covers the whole sequence. The
leg is located from the frame statistics: as the drone moves, real scene has high per-pixel
temporal variance while the static leg has near-zero variance. Classifying a pixel as leg
requires low temporal variance AND low intensity (the leg is near-black) AND connectivity to
an image border, the intensity and border tests reject blown-out sky (also low-variance) and
transient dark scene content.

Masks use 0 = ignore, 255 = keep, matching both COLMAP (feature_extractor --ImageReader.mask_path)
and nerfstudio (ColmapDataParser masks_path). Canonical masks are written to
<colmap>/masks/.canonical/camera{1..4}.png (cameras 1 and 3 are all-white). --materialize then
links a per-frame mask for every image under both naming schemes the two tools expect:

    masks/cameraX/imageNNNN.png        # nerfstudio: image suffix replaced with .png
    masks/cameraX/imageNNNN.jpeg.png   # COLMAP: .png appended to the full image name

Requires cv2 and scipy - run with the nerfstudio environment interpreter:

    python gen_leg_masks.py --overlay-dir DIR
    python gen_leg_masks.py --materialize

--overlay-dir writes a per-camera contact sheet (mask drawn over bright and dark sample
frames) for visual inspection before the masks are used in a reconstruction.
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

COLMAP_DIR = Path(
    "/home/dkhuttan/dataset/wrp_roof/before_solar_panels/all_cams_wrp_roof_10Jul2026_rgbs_colmap"
)
IMAGES_DIR = COLMAP_DIR / "images"
MASKS_DIR = COLMAP_DIR / "masks"
CANON_DIR = MASKS_DIR / ".canonical"

OBSTRUCTED = ["camera2", "camera4"]
CLEAN = ["camera1", "camera3"]

# Leg-detection parameters (intensities on a 0-255 grayscale).
N_SAMPLE = 180          # frames sampled for the temporal statistics
STD_THRESH = 15.0       # max per-pixel temporal std to count as static
DARK_THRESH = 90.0      # max temporal-mean intensity to count as leg (near-black)
CLOSE_KERNEL = 9        # closing radius to fill foot threads and specular holes
MIN_AREA_FRAC = 0.01    # discard static-dark regions below this fraction of the frame
DILATE_PX = 20          # margin added to absorb soft edges, motion blur, and vibration


def list_frames(cam_dir: Path):
    return sorted(cam_dir.glob("image*.jpeg"))


def sample_indices(n_total: int, n_sample: int):
    if n_total <= n_sample:
        return list(range(n_total))
    return list(np.linspace(0, n_total - 1, n_sample).round().astype(int))


def compute_leg_mask(cam_dir: Path):
    """Return (leg_bool [H,W] True=leg, mean_img, std_img) for one obstructed camera."""
    frames = list_frames(cam_dir)
    idxs = sample_indices(len(frames), N_SAMPLE)
    stack = []
    for i in idxs:
        g = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        stack.append(g.astype(np.float32))
    stack = np.stack(stack, axis=0)             # [N,H,W]
    mean_img = stack.mean(axis=0)
    std_img = stack.std(axis=0)

    candidate = (std_img < STD_THRESH) & (mean_img < DARK_THRESH)

    # Close small holes left by specular glints on the pole and remove speckle
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_KERNEL, CLOSE_KERNEL))
    candidate = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)

    # The leg enters from an image border, so keep every sufficiently large connected
    # component that touches an edge and discard interior specks
    lbl, n = ndimage.label(candidate)
    H, W = candidate.shape
    min_area = MIN_AREA_FRAC * H * W
    border_labels = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border_labels.discard(0)
    leg = np.zeros_like(candidate)
    for lb in border_labels:
        comp = lbl == lb
        if comp.sum() >= min_area:
            leg |= comp

    leg = ndimage.binary_fill_holes(leg)

    if DILATE_PX > 0:
        dk = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1)
        )
        leg = cv2.dilate(leg.astype(np.uint8), dk).astype(bool)

    return leg, mean_img, std_img


def save_canonical(cam: str, leg_bool):
    """Write black=leg=ignore (0), white=keep (255) as PNG and .npy"""
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    mask = np.where(leg_bool, 0, 255).astype(np.uint8)
    cv2.imwrite(str(CANON_DIR / f"{cam}.png"), mask)
    np.save(str(CANON_DIR/f"{cam}.npy"), mask)


def save_white(cam: str, shape):
    CANON_DIR.mkdir(parents=True, exist_ok=True)
    mask = np.full(shape, 255, np.uint8)
    cv2.imwrite(str(CANON_DIR / f"{cam}.png"), mask)
    np.save(str(CANON_DIR / f"{cam}.npy"), mask)


def write_overlays(cam: str, leg_bool, overlay_dir: Path):
    """Contact sheet: mask (red) over the brightest and darkest sample frames"""
    overlay_dir.mkdir(parents=True, exist_ok=True)
    frames = list_frames(IMAGES_DIR / cam)
    idxs = sample_indices(len(frames), N_SAMPLE)
    # Sort the sampled frames by mean brightness and pick a spread from darkest to brightest
    bright = [(cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE).mean(), i) for i in idxs]
    bright.sort()
    picks = [bright[0][1], bright[len(bright)//4][1], bright[len(bright)//2][1],
             bright[3*len(bright)//4][1], bright[-1][1]]
    tiles = []
    red = np.zeros((*leg_bool.shape, 3), np.uint8)
    red[leg_bool] = (0, 0, 255)  # BGR
    for i in picks:
        img = cv2.imread(str(frames[i]))
        ov = cv2.addWeighted(img, 1.0, red, 0.45, 0)
        # Outline the mask boundary so the exact masked region is visible
        cnts, _ = cv2.findContours(leg_bool.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 255), 2)
        tiles.append(ov)
    sheet = np.vstack(tiles)
    out = overlay_dir / f"{cam}_overlay.png"
    cv2.imwrite(str(out), sheet)
    frac = 100.0 * leg_bool.mean()
    print(f"  {cam}: leg covers {frac:.1f}% of frame  ->  {out}")


def materialize():
    """Create per-frame symlinks (both naming schemes) pointing at canonical masks"""
    n_links = 0
    for cam in CLEAN + OBSTRUCTED:
        src_rel = Path("..") / ".canonical" / f"{cam}.png"  # relative to masks/cameraX/
        canon = CANON_DIR / f"{cam}.png"
        assert canon.exists(), f"missing canonical mask {canon}; run without --materialize first"
        out_dir = MASKS_DIR / cam
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in list_frames(IMAGES_DIR / cam):
            stem = f.stem  # imageNNNN
            for name in (f"{stem}.png", f"{f.name}.png"):  # nerfstudio + COLMAP names
                link = out_dir / name
                if link.is_symlink() or link.exists():
                    link.unlink()
                os.symlink(src_rel, link)
                n_links += 1
    print(f"materialized {n_links} mask symlinks under {MASKS_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay-dir", type=Path, default=None,
                    help="write verification contact sheets here")
    ap.add_argument("--materialize", action="store_true",
                    help="build the masks/ symlink tree from existing canonical masks")
    args = ap.parse_args()

    if args.materialize:
        materialize()
        return

    shape = None
    for cam in OBSTRUCTED:
        print(f"[{cam}] computing temporal statistics over {N_SAMPLE} frames ...")
        leg, mean_img, std_img = compute_leg_mask(IMAGES_DIR / cam)
        shape = leg.shape
        save_canonical(cam, leg)
        if args.overlay_dir:
            write_overlays(cam, leg, args.overlay_dir)
    for cam in CLEAN:
        save_white(cam, shape)
        print(f"[{cam}] wrote all-white canonical mask")
    print(f"\ncanonical masks -> {CANON_DIR}")


if __name__ == "__main__":
    main()
