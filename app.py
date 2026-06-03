"""Streamlit dashboard for thin-section mineral analysis.

Upload a petrographic thin-section photo → SAM segments the grains → a classifier
labels each grain → you get the modal mineralogy plus per-grain uncertainty.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.classify import build_classifier
from src.interactive import build_mineral_figure
from src.minerals import resolve_preset
from src.pipeline import analyze_thin_section
from src.viz import draw_uncertainty_overlay, mineral_color_map

st.set_page_config(page_title="Thin-Section Mineral Analyzer", layout="wide")


@st.cache_resource(show_spinner="Loading model…")
def get_classifier(backend: str, preset: str):
    """Build (and cache) the classifier so weights load once, not every rerun."""
    if backend == "finetuned":
        return build_classifier("finetuned")
    return build_classifier(backend, minerals=resolve_preset(preset))


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Settings")
backend = st.sidebar.selectbox(
    "Classifier", ["clip", "finetuned"],
    help="clip = zero-shot, works on any rock (default). "
         "finetuned = trained CNN (5 granite minerals only; experimental).",
)
preset = st.sidebar.selectbox(
    "Mineral set", ["granite", "default", "ultramafic"],
    help="Candidate minerals for the clip backend. 'granite' = the 8 granite "
         "minerals with diagnostic XPL descriptions (recommended).",
    disabled=(backend == "finetuned"),
)
tta = st.sidebar.checkbox(
    "Test-time augmentation (uncertainty)", value=True,
    help="Classify several rotated/flipped views per grain and average them. "
         "Yields a stability score and steadier probabilities.",
)
max_entropy = st.sidebar.slider(
    "Uncertainty cutoff (max entropy)", 0.0, 1.0, 1.0, 0.05,
    help="Grains with normalized entropy ABOVE this are flagged 'uncertain'. "
         "1.0 = never flag. Lower it to be stricter.",
)
exclude_non_grain = st.sidebar.checkbox(
    "Exclude background / non-grain", value=True,
    help="Classify background, epoxy, holes and edges as 'non-grain' and leave "
         "them OUT of the modal mineralogy. (clip/claude backends only.)",
)
with st.sidebar.expander("Segmentation (advanced)"):
    min_area = st.number_input("Min grain area (px)", 50, 5000, 200, 50)
    points_per_side = st.slider("SAM grid density", 8, 32, 16, 4)
    long_edge = st.slider("Resize long edge (px)", 512, 2048, 1024, 256)
    bg_dark_frac = st.slider(
        "Background sensitivity", 0.3, 0.9, 0.6, 0.05,
        help="A grain is 'non-grain' if this fraction of its pixels are "
             "dark+unsaturated. Lower = exclude more as background.",
    )

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("🔬 Thin-Section Mineral Analyzer")
st.caption(
    "Upload a cross-polarized thin-section image. The app segments mineral "
    "grains, identifies each, reports the modal mineralogy (area % / count %), "
    "and quantifies per-grain uncertainty."
)

uploads = st.file_uploader(
    "Upload thin-section image(s)",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)

if not uploads:
    st.info("⬆️ Upload one or more thin-section images to begin.")
    st.stop()

classifier = get_classifier(backend, preset)

for upload in uploads:
    st.markdown("---")
    st.subheader(f"📄 {upload.name}")
    image_rgb = np.array(Image.open(upload).convert("RGB"))

    with st.spinner("Segmenting + classifying…"):
        result = analyze_thin_section(
            image_rgb,
            classifier=classifier,
            minerals=resolve_preset(preset),
            min_area=int(min_area),
            long_edge=int(long_edge),
            points_per_side=int(points_per_side),
            tta=tta,
            max_entropy=max_entropy,
            exclude_non_grain=exclude_non_grain,
            bg_dark_frac=bg_dark_frac,
        )

    summary = result.summary
    # Colormap over every predicted label so non-grain/uncertain also render.
    all_labels = [m.mineral for m in summary.minerals]
    all_labels += [l for l in dict.fromkeys(p.label for p in result.predictions)
                   if l not in all_labels]
    cmap = mineral_color_map(all_labels)

    # Headline metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mineral grains", result.n_grains - result.n_non_grain)
    top = summary.minerals[0].mineral if summary.minerals else "—"
    c2.metric("Dominant mineral (by area)", top)
    c3.metric("Flagged uncertain", f"{result.n_uncertain}")
    c4.metric("Non-grain excluded", f"{result.n_non_grain}")

    # Interactive mineral map — hover any grain to see its prediction.
    st.markdown("**Predicted minerals** — hover a grain for its label, "
                "confidence and uncertainty")
    fig = build_mineral_figure(result.image_rgb, result.grains,
                               result.predictions, cmap)
    st.plotly_chart(fig, use_container_width=True)

    # Static reference images
    i1, i2 = st.columns(2)
    i1.image(image_rgb, caption="Original", use_container_width=True)
    i2.image(
        draw_uncertainty_overlay(result.image_rgb, result.grains, result.predictions),
        caption="Uncertainty (green=sure, red=unsure)", use_container_width=True,
    )

    # Modal mineralogy table + chart
    modal_df = pd.DataFrame([
        {
            "mineral": m.mineral,
            "grains": m.n_grains,
            "count %": round(m.count_frac * 100, 1),
            "area %": round(m.area_frac * 100, 1),
            "mean conf": round(m.mean_confidence, 2),
            "mean entropy": round(m.mean_entropy, 2),
        }
        for m in summary.minerals
    ])
    t1, t2 = st.columns([3, 2])
    t1.markdown("**Modal mineralogy**")
    t1.dataframe(modal_df, use_container_width=True, hide_index=True)
    t2.markdown("**Area %**")
    t2.bar_chart(modal_df.set_index("mineral")["area %"])

    # Per-grain detail
    with st.expander(f"Per-grain detail ({result.n_grains} grains)"):
        grain_df = pd.DataFrame([
            {
                "grain": g.index,
                "mineral": p.label,
                "confidence": round(p.confidence, 3),
                "entropy": round(p.entropy, 3),
                "agreement": round(p.agreement, 3),
                "area_px": g.area_px,
            }
            for g, p in zip(result.grains, result.predictions)
        ])
        st.dataframe(grain_df, use_container_width=True, hide_index=True)

    # Download
    report = {
        "image": upload.name,
        "backend": backend,
        "n_grains": result.n_grains,
        "n_uncertain": result.n_uncertain,
        "modal_summary": summary.to_dict(),
    }
    st.download_button(
        "⬇️ Download summary.json",
        data=json.dumps(report, indent=2),
        file_name=f"{upload.name}_summary.json",
        mime="application/json",
    )
