"""CLI: segment grains in a thin-section image using SAM.

Usage:
    python segment.py path/to/image.jpg
    python segment.py path/to/image.jpg --model vit_b --min-area 500 --out outputs/run1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.segmentation import (
    draw_overlay,
    read_image_rgb,
    segment_grains,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path, help="Path to thin-section image")
    p.add_argument(
        "--model",
        choices=["vit_b", "vit_l", "vit_h"],
        default="vit_b",
        help="SAM backbone (vit_b is smallest / fastest)",
    )
    p.add_argument(
        "--min-area",
        type=int,
        default=200,
        help="Drop grains smaller than this many pixels",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: outputs/<image_stem>)",
    )
    args = p.parse_args()

    if not args.image.exists():
        p.error(f"Image not found: {args.image}")

    out_dir = args.out or Path("outputs") / args.image.stem
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"[segment] Reading {args.image}")
    image_rgb = read_image_rgb(args.image)
    h, w = image_rgb.shape[:2]
    print(f"[segment] Image {w}x{h}, running SAM ({args.model})...")

    grains = segment_grains(image_rgb, model_type=args.model, min_area=args.min_area)
    print(f"[segment] Found {len(grains)} grains (>= {args.min_area} px)")

    # Write per-grain masks + manifest.
    manifest = []
    for g in grains:
        mask_path = masks_dir / f"grain_{g.index:04d}.png"
        cv2.imwrite(str(mask_path), g.mask * 255)
        manifest.append(
            {
                "index": g.index,
                "mask": str(mask_path.relative_to(out_dir)),
                "bbox_xywh": list(g.bbox_xywh),
                "area_px": g.area_px,
                "predicted_iou": g.score,
            }
        )

    overlay_bgr = cv2.cvtColor(draw_overlay(image_rgb, grains), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "overlay.png"), overlay_bgr)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "image": str(args.image),
                "image_size": [w, h],
                "model": args.model,
                "min_area": args.min_area,
                "n_grains": len(grains),
                "grains": manifest,
            },
            indent=2,
        )
    )
    print(f"[segment] Wrote {out_dir / 'overlay.png'} and {len(grains)} masks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
