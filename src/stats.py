"""Aggregate per-grain predictions into a thin-section mineral distribution.

Reports both:
- **count %**  — fraction of grains assigned to each mineral.
- **area %**   — fraction of total segmented grain area per mineral. This is the
  petrographically meaningful "mode" (modal mineralogy): by the Delesse
  principle, area fraction in a random section estimates volume fraction.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .classify import Prediction
from .segmentation import Grain


@dataclass
class MineralStat:
    mineral: str
    n_grains: int
    count_frac: float
    area_px: int
    area_frac: float
    mean_confidence: float
    mean_entropy: float = 0.0


@dataclass
class ModalSummary:
    n_grains_total: int  # mineral grains only (excluded labels not counted)
    area_px_total: int   # mineral area only
    minerals: list[MineralStat]  # sorted by area_frac, descending
    # Non-grain / excluded categories (e.g. background), reported but NOT part
    # of the modal percentages.
    n_excluded: int = 0
    area_excluded_px: int = 0
    excluded_area_frac: float = 0.0  # excluded area / total segmented area

    def to_dict(self) -> dict:
        return {
            "n_grains_total": self.n_grains_total,
            "area_px_total": self.area_px_total,
            "minerals": [asdict(m) for m in self.minerals],
            "n_excluded": self.n_excluded,
            "area_excluded_px": self.area_excluded_px,
            "excluded_area_frac": self.excluded_area_frac,
        }


def summarize(
    grains: list[Grain],
    predictions: list[Prediction],
    exclude_labels: tuple[str, ...] = (),
) -> ModalSummary:
    """Aggregate predictions into modal mineralogy.

    Grains whose label is in ``exclude_labels`` (e.g. ``"non-grain"``) are kept
    OUT of the mineral percentages and reported separately — modal mineralogy is
    over actual mineral grains only.
    """
    if len(grains) != len(predictions):
        raise ValueError(
            f"grains ({len(grains)}) and predictions ({len(predictions)}) must align"
        )

    excluded = set(exclude_labels)
    total_area_all = sum(g.area_px for g in grains)

    # Partition into mineral grains vs excluded (non-grain) grains.
    kept = [(g, p) for g, p in zip(grains, predictions) if p.label not in excluded]
    dropped = [(g, p) for g, p in zip(grains, predictions) if p.label in excluded]

    n_total = len(kept)
    area_total = sum(g.area_px for g, _ in kept)

    agg: dict[str, dict[str, float]] = {}
    for g, p in kept:
        a = agg.setdefault(
            p.label, {"n": 0, "area": 0, "conf_sum": 0.0, "ent_sum": 0.0}
        )
        a["n"] += 1
        a["area"] += g.area_px
        a["conf_sum"] += p.confidence
        a["ent_sum"] += p.entropy

    stats = [
        MineralStat(
            mineral=label,
            n_grains=int(a["n"]),
            count_frac=a["n"] / n_total if n_total else 0.0,
            area_px=int(a["area"]),
            area_frac=a["area"] / area_total if area_total else 0.0,
            mean_confidence=a["conf_sum"] / a["n"] if a["n"] else 0.0,
            mean_entropy=a["ent_sum"] / a["n"] if a["n"] else 0.0,
        )
        for label, a in agg.items()
    ]
    stats.sort(key=lambda s: s.area_frac, reverse=True)

    area_excluded = sum(g.area_px for g, _ in dropped)
    return ModalSummary(
        n_grains_total=n_total,
        area_px_total=area_total,
        minerals=stats,
        n_excluded=len(dropped),
        area_excluded_px=area_excluded,
        excluded_area_frac=(area_excluded / total_area_all if total_area_all else 0.0),
    )


def format_table(summary: ModalSummary) -> str:
    """A compact text table for stdout."""
    lines = [
        f"{'mineral':<22}{'grains':>7}{'count%':>9}{'area%':>9}"
        f"{'mean conf':>11}{'mean entropy':>14}",
        "-" * 73,
    ]
    for s in summary.minerals:
        lines.append(
            f"{s.mineral:<22}{s.n_grains:>7}{s.count_frac * 100:>8.1f}%"
            f"{s.area_frac * 100:>8.1f}%{s.mean_confidence:>11.2f}{s.mean_entropy:>14.2f}"
        )
    lines.append("-" * 73)
    lines.append(
        f"{'TOTAL (minerals)':<22}{summary.n_grains_total:>7}{'100.0%':>9}{'100.0%':>9}"
    )
    if summary.n_excluded:
        lines.append(
            f"{'excluded non-grain':<22}{summary.n_excluded:>7}"
            f"{'':>9}{summary.excluded_area_frac * 100:>8.1f}%"
            f"   (of all segmented area; not in modal %)"
        )
    return "\n".join(lines)
