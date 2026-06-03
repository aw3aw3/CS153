"""CLI: full thin-section analysis — segment grains, classify each by mineral,
and report the mineral (modal) distribution.

Pipeline:  image -> SAM segmentation -> per-grain crops -> mineral classifier
           -> count% / area% summary + visual outputs

Examples:
    python analyze.py data\\my_thin_section.jpg
    python analyze.py img.jpg --minerals ultramafic --min-confidence 0.15
    python analyze.py img.jpg --backend claude          # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from src.classify import build_classifier
from src.minerals import resolve_preset
from src.pipeline import analyze_thin_section
from src.stats import format_table
from src.viz import draw_mineral_overlay, mineral_color_map, save_distribution_chart


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="Path to thin-section image")
    p.add_argument("--backend", choices=["clip", "claude", "finetuned"], default="clip",
                   help="Classifier backend (clip=local zero-shot default; "
                        "finetuned=trained CNN; claude=vision API)")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mineral_cnn.pt"),
                   help="Trained-model checkpoint for --backend finetuned")
    p.add_argument("--minerals", default="default",
                   help="Mineral preset: 'default' or 'ultramafic'")
    p.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], default="vit_b",
                   help="SAM backbone (default vit_b)")
    p.add_argument("--long-edge", type=int, default=1024,
                   help="Resize input so max(h,w)==this before SAM (0=disable)")
    p.add_argument("--points-per-side", type=int, default=16,
                   help="SAM auto-mask grid density (default 16)")
    p.add_argument("--min-area", type=int, default=200,
                   help="Drop grains smaller than this many original-res pixels")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Relabel predictions below this confidence as 'uncertain'")
    p.add_argument("--no-mask-bg", action="store_true",
                   help="Feed raw rectangular crops (don't grey out background)")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: outputs/<image_stem>_analysis)")
    args = p.parse_args()

    if not args.image.exists():
        p.error(f"Image not found: {args.image}")

    minerals = resolve_preset(args.minerals)
    out_dir = args.out or Path("outputs") / f"{args.image.stem}_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # long_edge<=0 disables the downscale (pass a value no image will exceed).
    long_edge = args.long_edge if args.long_edge > 0 else 1_000_000
    print(f"[analyze] {args.image}  backend={args.backend}  minerals={args.minerals}")

    # Build the classifier once (so weights load a single time).
    clf_kwargs = {}
    if args.backend == "finetuned":
        clf_kwargs["checkpoint"] = str(args.checkpoint)
    classifier = build_classifier(args.backend, minerals=minerals, **clf_kwargs)

    t0 = time.perf_counter()
    result = analyze_thin_section(
        args.image,
        classifier=classifier,
        minerals=minerals,
        model_type=args.model,
        min_area=args.min_area,
        long_edge=long_edge,
        points_per_side=args.points_per_side,
        mask_background=not args.no_mask_bg,
        min_confidence=args.min_confidence,
        progress=True,
    )
    elapsed = time.perf_counter() - t0

    summary = result.summary
    print()
    print(format_table(summary))
    print(f"\n[analyze] {result.n_grains} grains classified in {elapsed:.1f}s")

    # --- visual + data outputs ------------------------------------------------
    cmap = mineral_color_map([m.mineral for m in summary.minerals])
    overlay = draw_mineral_overlay(result.image_rgb, result.grains,
                                   result.predictions, cmap=cmap)
    cv2.imwrite(str(out_dir / "mineral_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    save_distribution_chart(summary, out_dir / "distribution.png", cmap=cmap)

    grains_json = [
        {
            "index": g.index,
            "bbox_xywh": list(g.bbox_xywh),
            "area_px": g.area_px,
            "mineral": pred.label,
            "confidence": round(pred.confidence, 4),
            "top3": dict(list(pred.scores.items())[:3]),
        }
        for g, pred in zip(result.grains, result.predictions)
    ]
    (out_dir / "summary.json").write_text(json.dumps({
        "image": str(args.image),
        "image_size": [result.image_rgb.shape[1], result.image_rgb.shape[0]],
        "backend": args.backend,
        "minerals_preset": args.minerals,
        "min_confidence": args.min_confidence,
        "elapsed_sec": round(elapsed, 2),
        "modal_summary": summary.to_dict(),
        "grains": grains_json,
    }, indent=2))

    print(f"[analyze] Wrote {out_dir}/ (mineral_overlay.png, distribution.png, summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
