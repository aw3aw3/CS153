"""Interactive (hover-able) grain overlay for the dashboard.

Builds a Plotly figure: the thin-section image with each grain drawn as a
filled polygon you can hover over to see its predicted mineral, confidence,
and uncertainty. Kept separate from ``viz.py`` (static matplotlib/cv2 outputs)
so the Plotly dependency stays in the UI layer.
"""
from __future__ import annotations

import cv2
import numpy as np

from .classify import Prediction
from .segmentation import Grain


def build_mineral_figure(
    image_rgb: np.ndarray,
    grains: list[Grain],
    predictions: list[Prediction],
    cmap: dict[str, tuple[int, int, int]],
    max_display: int = 900,
):
    """Return a Plotly Figure with one hoverable polygon per grain.

    Hovering a grain shows ``mineral · conf · entropy · agreement``. The image is
    downscaled to ``max_display`` on its long edge (with contour coords scaled to
    match) so the figure stays light in the browser.
    """
    import plotly.graph_objects as go

    h, w = image_rgb.shape[:2]
    scale = min(1.0, max_display / max(h, w))
    if scale < 1.0:
        disp = cv2.resize(
            image_rgb, (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        disp = image_rgb
    dh, dw = disp.shape[:2]

    fig = go.Figure()
    fig.add_trace(go.Image(z=disp, hoverinfo="skip"))

    for g, p in zip(grains, predictions):
        contours, _ = cv2.findContours(
            g.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea).squeeze()
        if c.ndim != 2 or len(c) < 3:
            continue
        xs = np.append(c[:, 0], c[0, 0]) * scale
        ys = np.append(c[:, 1], c[0, 1]) * scale
        r, gc, b = cmap.get(p.label, (127, 127, 127))
        label = (
            f"<b>{p.label}</b><br>conf {p.confidence:.2f} · "
            f"entropy {p.entropy:.2f} · agree {p.agreement:.2f}"
        )
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, fill="toself", mode="lines",
                line=dict(width=1, color="rgba(255,255,255,0.55)"),
                fillcolor=f"rgba({r},{gc},{b},0.35)",
                hoveron="fills", hoverinfo="text", text=label,
                name=p.label, showlegend=False,
            )
        )

    fig.update_xaxes(visible=False, showgrid=False, range=[0, dw])
    fig.update_yaxes(visible=False, showgrid=False, range=[dh, 0], scaleanchor="x")
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=620, hovermode="closest", dragmode="pan",
        plot_bgcolor="white",
    )
    return fig
