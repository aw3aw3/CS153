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
    n_grains_total: int
    area_px_total: int
    minerals: list[MineralStat]  # sorted by area_frac, descending

    def to_dict(self) -> dict:
        return {
            "n_grains_total": self.n_grains_total,
            "area_px_total": self.area_px_total,
            "minerals": [asdict(m) for m in self.minerals],
        }


def summarize(grains: list[Grain], predictions: list[Prediction]) -> ModalSummary:
    if len(grains) != len(predictions):
        raise ValueError(
            f"grains ({len(grains)}) and predictions ({len(predictions)}) must align"
        )

    n_total = len(grains)
    area_total = sum(g.area_px for g in grains)

    agg: dict[str, dict[str, float]] = {}
    for g, p in zip(grains, predictions):
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
    return ModalSummary(
        n_grains_total=n_total, area_px_total=area_total, minerals=stats
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
        f"{'TOTAL':<22}{summary.n_grains_total:>7}{'100.0%':>9}{'100.0%':>9}"
    )
    return "\n".join(lines)
