"""Mineral vocabulary and zero-shot prompt templates for thin-section classification.

The default vocabulary covers the common rock-forming minerals seen in
petrographic thin sections under cross-polarized light. Each entry has a
canonical ``name`` (used everywhere downstream) plus a list of ``aliases`` /
descriptive phrases that get expanded into CLIP text prompts. Editing this list
is the main lever for adapting the classifier to a particular rock type
(e.g. an ultramafic suite needs olivine/pyroxene/spinel/serpentine).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Mineral:
    name: str
    # Extra descriptive phrases fed to the text encoder alongside ``name``.
    aliases: tuple[str, ...] = field(default_factory=tuple)


# Default label set. Ordered roughly by how common they are in igneous /
# metamorphic thin sections. Keep names lowercase and singular.
DEFAULT_MINERALS: tuple[Mineral, ...] = (
    Mineral("quartz"),
    Mineral("plagioclase feldspar", ("plagioclase", "albite", "andesine")),
    Mineral("alkali feldspar", ("orthoclase", "microcline", "potassium feldspar")),
    Mineral("olivine", ("forsterite",)),
    Mineral("orthopyroxene", ("enstatite", "hypersthene")),
    Mineral("clinopyroxene", ("augite", "diopside")),
    Mineral("amphibole", ("hornblende",)),
    Mineral("biotite", ("biotite mica", "dark mica")),
    Mineral("muscovite", ("muscovite mica", "white mica")),
    Mineral("calcite", ("carbonate",)),
    Mineral("garnet", ("almandine", "pyrope")),
    Mineral("spinel", ("chromite",)),
    Mineral("serpentine", ("serpentinite",)),
    Mineral("chlorite", ()),
    Mineral("epidote", ()),
    Mineral("opaque mineral", ("magnetite", "ilmenite", "opaque oxide")),
)

# Ultramafic-focused preset (handy for the peridotite test image).
ULTRAMAFIC_MINERALS: tuple[Mineral, ...] = (
    Mineral("olivine", ("forsterite",)),
    Mineral("orthopyroxene", ("enstatite", "hypersthene")),
    Mineral("clinopyroxene", ("augite", "diopside")),
    Mineral("spinel", ("chromite",)),
    Mineral("serpentine", ("serpentinite",)),
    Mineral("amphibole", ("hornblende",)),
    Mineral("opaque mineral", ("magnetite", "ilmenite", "opaque oxide")),
)

PRESETS: dict[str, tuple[Mineral, ...]] = {
    "default": DEFAULT_MINERALS,
    "ultramafic": ULTRAMAFIC_MINERALS,
}

# CLIP zero-shot works best when each label's text embedding is the average of
# several phrasings. These templates are filled with each mineral name/alias.
PROMPT_TEMPLATES: tuple[str, ...] = (
    "a cross-polarized photomicrograph of {} in a petrographic thin section",
    "a thin section microscopy image of the mineral {}",
    "{} grain seen under a polarizing microscope",
    "photomicrograph of {} crystal in cross-polarized light",
)


def mineral_names(minerals: tuple[Mineral, ...] = DEFAULT_MINERALS) -> list[str]:
    return [m.name for m in minerals]


def build_prompts(
    minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
) -> tuple[list[str], list[list[str]]]:
    """Return ``(names, prompts_per_name)``.

    ``prompts_per_name[i]`` is the list of text prompts whose embeddings should
    be averaged to form the classifier weight for ``names[i]``.
    """
    names: list[str] = []
    prompts_per_name: list[list[str]] = []
    for m in minerals:
        phrases = (m.name, *m.aliases)
        prompts = [t.format(p) for p in phrases for t in PROMPT_TEMPLATES]
        names.append(m.name)
        prompts_per_name.append(prompts)
    return names, prompts_per_name


def resolve_preset(name: str) -> tuple[Mineral, ...]:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown mineral preset {name!r}. Options: {sorted(PRESETS)}"
        )
    return PRESETS[name]
