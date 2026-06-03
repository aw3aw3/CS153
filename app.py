"""Streamlit dashboard for thin-section mineral analysis.

Workflow: start a **New sample**, upload its series of stage-rotation photos,
and analyze. The pipeline segments grains (SAM), fuses the rotations into one
stronger per-grain prediction, and reports the modal mineralogy with
per-grain uncertainty.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.classify import build_classifier
from src.interactive import build_mineral_figure
from src.minerals import resolve_preset
from src.pipeline import analyze_thin_section_multi
from src.viz import draw_uncertainty_overlay, mineral_color_map

st.set_page_config(page_title="Thin-Section Mineral Analyzer", page_icon="🔬",
                   layout="wide")

# --- light, slick styling (visual only; no functional changes) -------------- #
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"], [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }
    h1 { font-weight: 700; letter-spacing: -0.6px; }
    h2, h3 { font-weight: 600; letter-spacing: -0.3px; }

    /* metric "cards" */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8ECF4;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stMetricLabel"] p { color: #64748B; font-weight: 500; }

    /* consistent, rounded buttons everywhere */
    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        border-radius: 10px;
        font-weight: 600;
        border: 1px solid #E2E8F0;
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.18);
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: #FAFBFE;
        border-right: 1px solid #EEF1F7;
    }
    [data-testid="stSidebar"] .stButton > button { width: 100%; text-align: left; }

    /* the New-sample form as a soft card */
    [data-testid="stForm"] {
        border: 1px solid #E8ECF4;
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        background: #FFFFFF;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }

    /* rounded tables, lighter chrome */
    [data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }
    hr { margin: 1.1rem 0; border-color: #EEF1F7; }
    </style>
    """,
    unsafe_allow_html=True,
)

UPLOAD_TYPES = ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]


@st.cache_resource(show_spinner="Loading model…")
def get_classifier(backend: str, preset: str):
    """Build (and cache) the classifier so weights load once, not every rerun."""
    if backend == "finetuned":
        return build_classifier("finetuned")
    return build_classifier(backend, minerals=resolve_preset(preset))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "samples" not in st.session_state:
    st.session_state.samples = []          # list of {name, result, backend, n_images}
if "view" not in st.session_state:
    st.session_state.view = "new"          # "new" or an int index into samples


def go_new():
    st.session_state.view = "new"


# --------------------------------------------------------------------------- #
# Sidebar — model settings + sample navigation
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🔬 Thin-Section Analyzer")
    st.caption("Granite mineral ID from stage-rotation photomicrographs")

    st.button("➕  New sample", use_container_width=True, type="primary",
              on_click=go_new)

    if st.session_state.samples:
        st.markdown("**Samples**")
        for i, s in enumerate(st.session_state.samples):
            label = f"📄 {s['name']}  ·  {s['n_images']} rot."
            st.button(label, key=f"nav_{i}", use_container_width=True,
                      on_click=lambda i=i: st.session_state.update(view=i))

    st.divider()
    st.markdown("**Model settings**")
    backend = st.selectbox(
        "Classifier", ["clip", "finetuned"],
        help="clip = zero-shot, works on any rock (default). "
             "finetuned = trained CNN (experimental, 5 granite minerals).",
    )
    preset = st.selectbox(
        "Mineral set", ["granite", "default", "ultramafic"],
        help="Candidate minerals for clip. 'granite' = the 8 granite minerals "
             "with diagnostic XPL descriptions (recommended).",
        disabled=(backend == "finetuned"),
    )
    tta = st.checkbox("Test-time augmentation", value=True,
                      help="Average several rotated/flipped views per grain.")
    max_entropy = st.slider(
        "Uncertainty cutoff (max entropy)", 0.0, 1.0, 1.0, 0.05,
        help="Grains with entropy above this are flagged 'uncertain'. 1.0 = off.",
    )
    exclude_non_grain = st.checkbox(
        "Exclude background / non-grain", value=True,
        help="Detect dark isotropic background/epoxy/holes and leave them out "
             "of the modal mineralogy.",
    )
    with st.expander("Advanced (segmentation)"):
        min_area = st.number_input("Min grain area (px)", 50, 5000, 200, 50)
        points_per_side = st.slider("SAM grid density", 8, 32, 16, 4)
        long_edge = st.slider("Resize long edge (px)", 512, 2048, 1024, 256)
        bg_dark_frac = st.slider(
            "Background sensitivity", 0.3, 0.9, 0.6, 0.05,
            help="Lower = exclude more as background.",
        )


def analysis_kwargs() -> dict:
    return dict(
        minerals=resolve_preset(preset), min_area=int(min_area),
        long_edge=int(long_edge), points_per_side=int(points_per_side),
        tta=tta, max_entropy=max_entropy, exclude_non_grain=exclude_non_grain,
        bg_dark_frac=bg_dark_frac,
    )


# --------------------------------------------------------------------------- #
# Result rendering
# --------------------------------------------------------------------------- #
def render_result(idx: int, record: dict) -> None:
    result = record["result"]
    summary = result.summary

    st.subheader(f"📄 {record['name']}")
    src = (f"Fused from {result.n_views} stage rotations"
           if result.n_views > 1 else "Single image")
    st.caption(f"{src}  ·  classifier: {record['backend']}")

    all_labels = [m.mineral for m in summary.minerals]
    all_labels += [l for l in dict.fromkeys(p.label for p in result.predictions)
                   if l not in all_labels]
    cmap = mineral_color_map(all_labels)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mineral grains", result.n_grains - result.n_non_grain)
    top = summary.minerals[0].mineral if summary.minerals else "—"
    c2.metric("Dominant mineral (area)", top)
    c3.metric("Flagged uncertain", f"{result.n_uncertain}")
    c4.metric("Non-grain excluded", f"{result.n_non_grain}")

    agree_note = ("  ·  *agreement = cross-rotation consensus*"
                  if result.n_views > 1 else "")
    st.markdown("**Predicted minerals** — hover a grain for its label, "
                "confidence and uncertainty" + agree_note)
    fig = build_mineral_figure(result.image_rgb, result.grains,
                               result.predictions, cmap)
    st.plotly_chart(fig, use_container_width=True, key=f"fig_{idx}")

    i1, i2 = st.columns(2)
    i1.image(result.image_rgb,
             caption="Reference view" if result.n_views > 1 else "Original",
             use_container_width=True)
    i2.image(
        draw_uncertainty_overlay(result.image_rgb, result.grains, result.predictions),
        caption="Uncertainty (green = sure, red = unsure)", use_container_width=True,
    )

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
    if not modal_df.empty:
        t2.bar_chart(modal_df.set_index("mineral")["area %"])

    with st.expander(f"Per-grain detail ({result.n_grains} grains)"):
        grain_df = pd.DataFrame([
            {
                "grain": g.index, "mineral": p.label,
                "confidence": round(p.confidence, 3),
                "entropy": round(p.entropy, 3),
                "agreement": round(p.agreement, 3),
                "area_px": g.area_px,
            }
            for g, p in zip(result.grains, result.predictions)
        ])
        st.dataframe(grain_df, use_container_width=True, hide_index=True)

    report = {
        "sample": record["name"], "backend": record["backend"],
        "n_rotations": result.n_views, "n_grains": result.n_grains,
        "n_uncertain": result.n_uncertain, "n_non_grain": result.n_non_grain,
        "modal_summary": summary.to_dict(),
    }
    st.download_button(
        "⬇️  Download summary.json", data=json.dumps(report, indent=2),
        file_name=f"{record['name']}_summary.json", mime="application/json",
        key=f"dl_{idx}",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if st.session_state.view == "new":
    st.title("🔬 Thin-Section Mineral Analyzer")
    st.markdown(
        "Identify the minerals in a thin section and measure its "
        "modal mineralogy with per-grain uncertainty."
    )
    a, b, c = st.columns(3)
    a.markdown("**1 · New sample**\nName it and upload its photos.")
    b.markdown("**2 · Stage rotations**\nAdd several views of the *same* field "
               "at different stage angles — they're fused for a stronger call.")
    c.markdown("**3 · Analyze**\nGet the mineral map, modal %, and uncertainty.")

    st.divider()
    with st.form("new_sample"):
        st.markdown("### ➕ New sample")
        name = st.text_input("Sample name",
                             value=f"Sample {len(st.session_state.samples) + 1}")
        files = st.file_uploader(
            "Stage-rotation images (one or more, same field of view)",
            type=UPLOAD_TYPES, accept_multiple_files=True,
        )
        align = st.checkbox("Align rotations before fusing", value=True,
                            help="ECC registration onto the first image.")
        submitted = st.form_submit_button("🔬  Analyze sample", type="primary",
                                          use_container_width=True)

    if submitted:
        if not files:
            st.error("Please upload at least one image.")
        else:
            arrays = [np.array(Image.open(f).convert("RGB")) for f in files]
            clf = get_classifier(backend, preset)
            with st.spinner(f"Segmenting + classifying {len(arrays)} rotation(s)…"):
                res = analyze_thin_section_multi(
                    arrays, classifier=clf, align=align, **analysis_kwargs(),
                )
            st.session_state.samples.append({
                "name": name or f"Sample {len(st.session_state.samples) + 1}",
                "result": res, "backend": backend, "n_images": len(arrays),
            })
            st.session_state.view = len(st.session_state.samples) - 1
            st.rerun()
else:
    idx = st.session_state.view
    render_result(idx, st.session_state.samples[idx])
