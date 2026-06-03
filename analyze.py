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
from src.viz import (
    draw_mineral_overlay,
    draw_uncertainty_overlay,
    mineral_color_map,
    save_distribution_chart,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image", type=Path, help="Path to thin-section image")
    p.add_argument("--backend", choices=["clip", "claude", "finetuned"], default="clip",
                   help="Classifier backend (clip=local zero-shot default; "
                        "finetuned=trained CNN; claude=vision API)")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/mineral_cnn.pt"),
                   help="Trained-model checkpoint for --backend finetuned")
    p.add_argument("--minerals", default="granite",
                   help="Mineral preset: 'granite' (default), 'default', 'ultramafic'")
    p.add_argument("--model", choices=["vit_b", "vit_l", "vit_h"], default="vit_b",
                   help="SAM backbone (default vit_b)")
    p.add_argument("--long-edge", type=int, default=1024,
                   help="Resize input so max(h,w)==this before SAM (0=disable)")
    p.add_argument("--points-per-side", type=int, default=16,
                   help="SAM auto-mask grid density (default 16)")
    p.add_argument("--min-area", type=int, default=200,
                   help="Drop grains smaller than this many original-res pixels")
    p.add_argument("--no-mask-bg", action="store_true",
                   help="Feed raw rectangular crops (don't grey out background)")
    # --- uncertainty ---
    p.add_argument("--tta", dest="tta", action="store_true", default=True,
                   help="Test-time augmentation (rotations/flips). On by default.")
    p.add_argument("--no-tta", dest="tta", action="store_false",
                   help="Disable TTA (faster, but no agreement score)")
    p.add_argument("--min-confidence", type=float, default=0.0,
                   help="Flag grains below this top-probability as 'uncertain'")
    p.add_argument("--max-entropy", type=float, default=1.0,
                   help="Flag grains with normalized entropy above this (0-1)")
    p.add_argument("--min-agreement", type=float, default=0.0,
                   help="Flag grains whose TTA view-agreement is below this (0-1)")
    p.add_argument("--keep-non-grain", action="store_true",
                   help="Keep background/epoxy/holes in the modal % "
                        "(default: detect + exclude them as 'non-grain')")
    p.add_argument("--bg-dark-frac", type=float, default=0.6,
                   help="A grain is 'non-grain' if at least this fraction of its "
                        "pixels are dark+unsaturated (0-1). Lower = stricter.")
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
        tta=args.tta,
        min_confidence=args.min_confidence,
        max_entropy=args.max_entropy,
        min_agreement=args.min_agreement,
        exclude_non_grain=not args.keep_non_grain,
        bg_dark_frac=args.bg_dark_frac,
        progress=True,
    )
    elapsed = time.perf_counter() - t0

    summary = result.summary
    print()
    print(format_table(summary))
    n_unc = result.n_uncertain
    print(f"\n[analyze] {result.n_grains} grains classified in {elapsed:.1f}s "
          f"(TTA={'on' if args.tta else 'off'}; {n_unc} uncertain; "
          f"{result.n_non_grain} non-grain excluded)")

    # --- visual + data outputs ------------------------------------------------
    # Build the colormap over EVERY predicted label (minerals + uncertain +
    # non-grain) so excluded categories still render in the overlay.
    all_labels = [m.mineral for m in summary.minerals]
    all_labels += [l for l in dict.fromkeys(p.label for p in result.predictions)
                   if l not in all_labels]
    cmap = mineral_color_map(all_labels)
    overlay = draw_mineral_overlay(result.image_rgb, result.grains,
                                   result.predictions, cmap=cmap)
    cv2.imwrite(str(out_dir / "mineral_overlay.png"),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    unc_overlay = draw_uncertainty_overlay(result.image_rgb, result.grains,
                                           result.predictions)
    cv2.imwrite(str(out_dir / "uncertainty_overlay.png"),
                cv2.cvtColor(unc_overlay, cv2.COLOR_RGB2BGR))
    save_distribution_chart(summary, out_dir / "distribution.png", cmap=cmap)

    grains_json = [
        {
            "index": g.index,
            "bbox_xywh": list(g.bbox_xywh),
            "area_px": g.area_px,
            "mineral": pred.label,
            "confidence": round(pred.confidence, 4),
            "entropy": round(pred.entropy, 4),
            "margin": round(pred.margin, 4),
            "agreement": round(pred.agreement, 4),
            "top3": {k: round(v, 4) for k, v in list(pred.scores.items())[:3]},
        }
        for g, pred in zip(result.grains, result.predictions)
    ]
    (out_dir / "summary.json").write_text(json.dumps({
        "image": str(args.image),
        "image_size": [result.image_rgb.shape[1], result.image_rgb.shape[0]],
        "backend": args.backend,
        "minerals_preset": args.minerals,
        "tta": args.tta,
        "uncertainty_thresholds": {
            "min_confidence": args.min_confidence,
            "max_entropy": args.max_entropy,
            "min_agreement": args.min_agreement,
        },
        "n_uncertain": n_unc,
        "n_non_grain_excluded": result.n_non_grain,
        "elapsed_sec": round(elapsed, 2),
        "modal_summary": summary.to_dict(),
        "grains": grains_json,
    }, indent=2))

    print(f"[analyze] Wrote {out_dir}/ (mineral_overlay.png, "
          f"uncertainty_overlay.png, distribution.png, summary.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
