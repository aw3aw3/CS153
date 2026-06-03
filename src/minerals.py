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
    # Extra short synonyms fed to the text encoder alongside ``name``.
    aliases: tuple[str, ...] = field(default_factory=tuple)
    # Full descriptive XPL captions (twinning, extinction, interference colors).
    # These are the strongest zero-shot signal — CLIP matches images to text, so
    # describing the diagnostic appearance beats a bare mineral name.
    descriptions: tuple[str, ...] = field(default_factory=tuple)


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

# Granite-focused preset (8 minerals) with diagnostic XPL descriptions.
# This is the primary target rock type. Descriptions encode the features a
# petrographer uses to tell these apart under crossed polars.
GRANITE_MINERALS: tuple[Mineral, ...] = (
    Mineral(
        "quartz",
        descriptions=(
            "quartz with first-order gray and white interference colors and wavy undulose extinction",
            "a clear quartz grain with low gray birefringence and no twinning under crossed polars",
        ),
    ),
    Mineral(
        "plagioclase",
        ("plagioclase feldspar",),
        descriptions=(
            "plagioclase feldspar with polysynthetic albite twinning forming alternating black and white parallel stripes",
            "plagioclase showing zebra-stripe lamellar twinning and first-order gray interference colors",
        ),
    ),
    Mineral(
        "microcline",
        ("microcline feldspar", "alkali feldspar"),
        descriptions=(
            "microcline alkali feldspar with cross-hatched tartan twinning resembling a plaid pattern",
            "microcline showing distinctive grid twinning and first-order gray colors",
        ),
    ),
    Mineral(
        "orthoclase",
        ("orthoclase feldspar", "alkali feldspar"),
        descriptions=(
            "orthoclase alkali feldspar with simple Carlsbad twinning dividing the crystal into two domains",
            "orthoclase showing first-order gray interference colors and a single Carlsbad twin boundary",
        ),
    ),
    Mineral(
        "biotite",
        ("biotite mica", "brown mica"),
        descriptions=(
            "brown biotite mica with high-order interference colors masked by strong brown body color",
            "biotite showing one cleavage, parallel extinction, and a mottled bird's-eye texture near extinction",
        ),
    ),
    Mineral(
        "muscovite",
        ("muscovite mica", "white mica"),
        descriptions=(
            "muscovite mica with bright high-order pink, blue, green and yellow interference colors",
            "colorless muscovite showing vivid second to third-order birefringence and bird's-eye extinction",
        ),
    ),
    Mineral(
        "hornblende",
        ("amphibole",),
        descriptions=(
            "green to brown hornblende amphibole with two cleavages intersecting at 56 and 124 degrees and inclined extinction",
            "hornblende showing upper first-order interference colors partly masked by green-brown pleochroism",
        ),
    ),
    Mineral(
        "zircon",
        ("zircon crystal",),
        descriptions=(
            "a tiny high-relief zircon crystal with extreme third to fourth-order interference colors",
            "a small zircon inclusion inside biotite surrounded by a dark pleochroic halo",
        ),
    ),
)

PRESETS: dict[str, tuple[Mineral, ...]] = {
    "granite": GRANITE_MINERALS,
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

# Wraps a full diagnostic description into an image-caption prompt for CLIP.
DESCRIPTION_TEMPLATE = "a cross-polarized thin-section photomicrograph of {}"


# "Non-grain" catch-all: things SAM segments that are NOT mineral grains —
# background, mounting epoxy, holes/cracks, image edges. Giving the zero-shot
# classifier explicit prompts for these lets it route such crops here instead of
# forcing them into a mineral class. Downstream they're excluded from the modal
# mineralogy. Prompts share a "no crystal / empty / dark" theme for a clean
# CLIP text centroid.
NON_GRAIN_LABEL = "non-grain"
NON_GRAIN_PROMPTS: tuple[str, ...] = (
    "a dark isotropic background with no mineral crystal",
    "plain black empty space under cross-polarized light",
    "epoxy resin or mounting medium, not a mineral",
    "an empty hole, crack, or void in the thin section",
    "a blank featureless region of the slide",
)


def mineral_names(minerals: tuple[Mineral, ...] = DEFAULT_MINERALS) -> list[str]:
    return [m.name for m in minerals]


def build_prompts(
    minerals: tuple[Mineral, ...] = DEFAULT_MINERALS,
    include_non_grain: bool = False,
) -> tuple[list[str], list[list[str]]]:
    """Return ``(names, prompts_per_name)``.

    ``prompts_per_name[i]`` is the list of text prompts whose embeddings should
    be averaged to form the classifier weight for ``names[i]``. When
    ``include_non_grain`` is set, a trailing ``NON_GRAIN_LABEL`` class is added.
    """
    names: list[str] = []
    prompts_per_name: list[list[str]] = []
    for m in minerals:
        phrases = (m.name, *m.aliases)
        prompts = [t.format(p) for p in phrases for t in PROMPT_TEMPLATES]
        # Diagnostic descriptions (twinning/extinction/colors) — the strongest cues.
        prompts += [DESCRIPTION_TEMPLATE.format(d) for d in m.descriptions]
        names.append(m.name)
        prompts_per_name.append(prompts)
    if include_non_grain:
        names.append(NON_GRAIN_LABEL)
        prompts_per_name.append(list(NON_GRAIN_PROMPTS))
    return names, prompts_per_name


def resolve_preset(name: str) -> tuple[Mineral, ...]:
    if name not in PRESETS:
        raise ValueError(
            f"Unknown mineral preset {name!r}. Options: {sorted(PRESETS)}"
        )
    return PRESETS[name]
