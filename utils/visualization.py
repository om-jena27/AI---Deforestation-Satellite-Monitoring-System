"""
utils/visualization.py
=======================
Visualization helpers for the deforestation dashboard.

All functions return numpy arrays (uint8 RGB) or matplotlib figures
suitable for st.image() / st.pyplot().
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# 1. Deforestation overlay
# ---------------------------------------------------------------------------

def create_deforestation_overlay(
    rgb_img: np.ndarray,
    deforest_mask: np.ndarray,
    opacity: float = 0.6,
) -> np.ndarray:
    """
    Blend a semi-transparent red mask over the original satellite image.

    Parameters
    ----------
    rgb_img       : (H, W, 3) uint8
    deforest_mask : (H, W) uint8  {0, 1}
    opacity       : alpha of the red overlay (0=invisible, 1=fully opaque)

    Returns
    -------
    (H, W, 3) uint8 composite image
    """
    if rgb_img.shape[:2] != deforest_mask.shape:
        deforest_mask = cv2.resize(
            deforest_mask, (rgb_img.shape[1], rgb_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = rgb_img.copy().astype(np.float32)
    red_layer = np.zeros_like(overlay)
    red_layer[:, :, 0] = 255  # pure red

    mask_bool = deforest_mask.astype(bool)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - opacity) + red_layer[mask_bool] * opacity
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. NDVI / probability colourmap
# ---------------------------------------------------------------------------

def create_ndvi_colormap(ndvi_arr: np.ndarray) -> np.ndarray:
    """
    Convert a float array (any range) to an RGB heatmap using RdYlGn colormap.
    Suitable for NDVI, probability maps, etc.

    Returns (H, W, 3) uint8.
    """
    norm = ndvi_arr.astype(np.float64)
    mn, mx = norm.min(), norm.max()
    if mx > mn:
        norm = (norm - mn) / (mx - mn)
    else:
        norm = np.zeros_like(norm)

    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap(norm)                           # (H, W, 4)
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8)
    return rgb


# ---------------------------------------------------------------------------
# 3. Confidence / probability heatmap (matplotlib figure)
# ---------------------------------------------------------------------------

def create_confidence_heatmap_figure(
    prob_map: np.ndarray,
    title: str = "Model Confidence Heatmap",
) -> Figure:
    """
    Create a matplotlib Figure showing the probability/confidence heatmap.

    Parameters
    ----------
    prob_map : (H, W) float32  [0..1]
    title    : figure title

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(prob_map, cmap="plasma", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Confidence / Probability", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Forest mask display (green = forest, dark = non-forest)
# ---------------------------------------------------------------------------

def create_forest_mask_display(forest_mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary forest mask to a green/dark RGB image for display.

    Parameters
    ----------
    forest_mask : (H, W) uint8  {0, 1}

    Returns
    -------
    (H, W, 3) uint8  — green where forest, near-black where non-forest
    """
    h, w = forest_mask.shape
    display = np.zeros((h, w, 3), dtype=np.uint8)

    forest_bool = forest_mask.astype(bool)
    display[forest_bool]  = [34, 139, 34]   # Forest Green
    display[~forest_bool] = [20,  20,  20]  # Near-black

    return display


# ---------------------------------------------------------------------------
# 5. Change map (4-class colour-coded)
# ---------------------------------------------------------------------------

def create_change_map(
    mask_t1: np.ndarray,
    mask_t2: np.ndarray,
) -> np.ndarray:
    """
    Colour-coded 4-class change map:
        Dark grey  — no forest at either time
        Green      — retained forest
        Red        — deforestation (forest → non-forest)
        Light blue — new vegetation (non-forest → forest)

    Returns (H, W, 3) uint8.
    """
    m1 = mask_t1.astype(bool)
    m2 = mask_t2.astype(bool)

    h, w = mask_t1.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    out[~m1 & ~m2] = [40,  40,  40]   # no forest
    out[m1 &  m2]  = [34, 139,  34]   # retained forest
    out[m1 & ~m2]  = [220,  30,  30]  # deforestation (red)
    out[~m1 & m2]  = [100, 180, 240]  # new vegetation (blue)

    return out


# ---------------------------------------------------------------------------
# 6. KPI bar chart
# ---------------------------------------------------------------------------

def create_kpi_bar_chart(area_metrics: dict) -> Figure:
    """
    Small bar chart showing forest T1 vs T2 pixel counts (or ha if available).
    """
    if area_metrics.get("geo_available"):
        labels = ["Forest T1", "Forest T2", "Deforested"]
        values = [
            area_metrics["forest_t1_ha"] or 0,
            area_metrics["forest_t2_ha"] or 0,
            area_metrics["deforest_ha"] or 0,
        ]
        unit = "Hectares"
    else:
        labels = ["Forest T1", "Forest T2", "Deforested"]
        values = [
            area_metrics["forest_t1_pixels"],
            area_metrics["forest_t2_pixels"],
            area_metrics["deforest_pixels"],
        ]
        unit = "Pixels"

    colors = ["#2ecc71", "#27ae60", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(6, 3.5), facecolor="#0e1117")
    ax.set_facecolor("#161b22")
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{val:,.0f}",
            ha="center", va="bottom",
            color="white", fontsize=9,
        )

    ax.set_ylabel(unit, color="white", fontsize=10)
    ax.set_title("Forest Area Comparison", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555")
    ax.spines["bottom"].set_color("#555")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Interactive Detection Map (Plotly Image-Coordinate Viewer)
# ---------------------------------------------------------------------------

def create_interactive_detection_map(
    rgb_img: np.ndarray,
    deforest_mask: np.ndarray,
    regions: list[dict] | None = None,
    pixel_res_m: float | None = None,
):
    """
    Creates an interactive Plotly map in image coordinates.
    Allows zooming, panning, and hovering over individual deforestation zones.
    Clearly labelled as image-coordinate space when no GPS/CRS is embedded.
    """
    import plotly.graph_objects as go
    from PIL import Image as PILImage

    h, w = rgb_img.shape[:2]

    # Resize if image is huge to keep browser silky smooth
    max_dim = 600
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        disp_w = int(w * scale)
        disp_h = int(h * scale)
        rgb_disp = cv2.resize(rgb_img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    else:
        disp_w, disp_h = w, h
        rgb_disp = rgb_img

    pil_img = PILImage.fromarray(rgb_disp)

    fig = go.Figure()

    # Base image layer
    fig.add_trace(
        go.Image(z=pil_img, hoverinfo="none")
    )

    # Add deforestation region bounding boxes and hover markers
    if regions:
        scale_x = disp_w / w
        scale_y = disp_h / h

        x_coords = []
        y_coords = []
        hover_texts = []
        custom_data = []

        for r in regions:
            cx_scaled = r["centroid"]["cx"] * scale_x
            cy_scaled = r["centroid"]["cy"] * scale_y
            x_coords.append(cx_scaled)
            y_coords.append(cy_scaled)

            conf_str = f"{r['confidence_pct']:.1f}%" if r.get("confidence_pct") is not None else "Estimated"
            area_str = r.get("area_str", f"{r['area_pixels']:,} px")

            hover_text = (
                f"<b>Zone #{r['id']:02d}</b><br>"
                f"• Area: {area_str}<br>"
                f"• Confidence: {conf_str}<br>"
                f"• Centroid: ({r['centroid']['cx']:.0f}, {r['centroid']['cy']:.0f}) px<br>"
                f"• BBox: {r['bbox']['w']} × {r['bbox']['h']} px"
            )
            hover_texts.append(hover_text)

            # Draw bounding box for top 25 largest zones
            if r["id"] <= 25:
                bx0 = r["bbox"]["x"] * scale_x
                by0 = r["bbox"]["y"] * scale_y
                bx1 = bx0 + r["bbox"]["w"] * scale_x
                by1 = by0 + r["bbox"]["h"] * scale_y
                fig.add_shape(
                    type="rect",
                    x0=bx0, y0=by0, x1=bx1, y1=by1,
                    line=dict(color="#FF3366", width=2, dash="dot"),
                    fillcolor="rgba(255, 50, 50, 0.25)",
                )

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                marker=dict(
                    size=10,
                    color="#FF0033",
                    symbol="circle",
                    line=dict(color="#FFFFFF", width=1.5),
                ),
                text=[f"#{r['id']:02d}" for r in regions],
                textposition="top center",
                textfont=dict(color="#FFDD55", size=10),
                hovertext=hover_texts,
                hoverinfo="text",
                name="Deforestation Zones",
            )
        )

    fig.update_layout(
        title=dict(
            text="Interactive Deforestation Zone Explorer (Zoom & Pan)",
            font=dict(color="#10B981", size=14),
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, visible=True,
            title="Image X (Pixels)", color="#9CA3AF"
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, visible=True,
            title="Image Y (Pixels)", autorange="reversed", color="#9CA3AF"
        ),
        margin=dict(l=40, r=40, t=50, b=40),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        height=520,
        legend=dict(font=dict(color="#FFFFFF")),
    )
    return fig


# ---------------------------------------------------------------------------
# 8. Geographic Map (Folium) for Georeferenced Coordinates
# ---------------------------------------------------------------------------

def create_folium_geographic_map(
    lat: float,
    lon: float,
    zoom_start: int = 11,
    site_name: str = "Deforestation Case Study",
    deforest_ha: float | None = None,
    risk_label: str = "MODERATE",
):
    """
    Renders an interactive Folium satellite map when lat/lon coordinates are available.
    """
    import folium

    m = folium.Map(
        location=[lat, lon],
        zoom_start=zoom_start,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
    )

    # Popup text
    popup_text = f"<b>{site_name}</b><br>Coordinates: {lat:.4f}, {lon:.4f}<br>Risk: {risk_label}"
    if deforest_ha is not None:
        popup_text += f"<br>Detected Loss: {deforest_ha:,.2f} ha"

    folium.Marker(
        [lat, lon],
        popup=popup_text,
        tooltip=f"{site_name} ({risk_label})",
        icon=folium.Icon(color="red", icon="warning-sign"),
    ).add_to(m)

    # Region perimeter indicator
    folium.Circle(
        radius=3000,
        location=[lat, lon],
        color="#e74c3c",
        fill=True,
        fill_color="#e74c3c",
        fill_opacity=0.2,
        tooltip="Monitored Area of Interest",
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# 9. Circular Gauge Charts (Plotly Glassmorphic Gauges)
# ---------------------------------------------------------------------------

def create_circular_gauges(
    loss_pct: float,
    retained_pct: float,
    confidence_pct: float,
    risk_level: str = "MODERATE",
) -> Any:
    """
    Renders 3 synchronized circular gauge meters (Plotly):
    1. Canopy Retained (%)
    2. Deforestation Canopy Loss (%)
    3. AI Model Confidence (%)

    Styled with dark theme, glassmorphic palette, and neon glow accents.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.07,
    )

    clamped_retained = float(np.clip(retained_pct, 0.0, 100.0))
    clamped_loss = float(np.clip(loss_pct, 0.0, 100.0))
    clamped_conf = float(np.clip(confidence_pct, 0.0, 100.0))

    # 1. Canopy Retained Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=clamped_retained,
            number={
                "suffix": "%",
                "font": {"size": 26, "color": "#10B981"},
            },
            title={
                "text": "<b>Canopy Retained</b><br><span style='font-size:11px;color:#9CA3AF'>Intact Forest Area</span>",
                "font": {"size": 13, "color": "#F3F4F6"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#4B5563",
                    "tickfont": {"size": 9, "color": "#9CA3AF"},
                },
                "bar": {"color": "#10B981", "thickness": 0.28},
                "bgcolor": "rgba(31, 41, 55, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239, 68, 68, 0.18)"},
                    {"range": [50, 80], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [80, 100], "color": "rgba(16, 185, 129, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": "#34D399", "width": 3},
                    "thickness": 0.75,
                    "value": clamped_retained,
                },
            },
        ),
        row=1,
        col=1,
    )

    # 2. Deforestation Loss Rate Gauge
    if clamped_loss < 2.0:
        loss_color = "#10B981"
    elif clamped_loss < 10.0:
        loss_color = "#F59E0B"
    else:
        loss_color = "#EF4444"

    loss_max_range = max(25.0, min(100.0, float((int(clamped_loss // 10) + 2) * 10)))

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=clamped_loss,
            number={
                "suffix": "%",
                "font": {"size": 26, "color": loss_color},
            },
            title={
                "text": f"<b>Deforestation Loss</b><br><span style='font-size:11px;color:#9CA3AF'>Risk: {risk_level}</span>",
                "font": {"size": 13, "color": "#F3F4F6"},
            },
            gauge={
                "axis": {
                    "range": [0, loss_max_range],
                    "tickwidth": 1,
                    "tickcolor": "#4B5563",
                    "tickfont": {"size": 9, "color": "#9CA3AF"},
                },
                "bar": {"color": loss_color, "thickness": 0.28},
                "bgcolor": "rgba(31, 41, 55, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 5], "color": "rgba(16, 185, 129, 0.18)"},
                    {"range": [5, 15], "color": "rgba(245, 158, 11, 0.18)"},
                    {"range": [15, loss_max_range], "color": "rgba(239, 68, 68, 0.22)"},
                ],
                "threshold": {
                    "line": {"color": loss_color, "width": 3},
                    "thickness": 0.75,
                    "value": clamped_loss,
                },
            },
        ),
        row=1,
        col=2,
    )

    # 3. Model Detection Confidence Gauge
    conf_color = "#38BDF8"
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=clamped_conf,
            number={
                "suffix": "%",
                "font": {"size": 26, "color": conf_color},
            },
            title={
                "text": "<b>AI Model Confidence</b><br><span style='font-size:11px;color:#9CA3AF'>Inference Certainty</span>",
                "font": {"size": 13, "color": "#F3F4F6"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#4B5563",
                    "tickfont": {"size": 9, "color": "#9CA3AF"},
                },
                "bar": {"color": conf_color, "thickness": 0.28},
                "bgcolor": "rgba(31, 41, 55, 0.5)",
                "borderwidth": 1,
                "bordercolor": "rgba(255, 255, 255, 0.1)",
                "steps": [
                    {"range": [0, 60], "color": "rgba(100, 116, 139, 0.18)"},
                    {"range": [60, 85], "color": "rgba(56, 189, 248, 0.18)"},
                    {"range": [85, 100], "color": "rgba(14, 165, 233, 0.26)"},
                ],
                "threshold": {
                    "line": {"color": conf_color, "width": 3},
                    "thickness": 0.75,
                    "value": clamped_conf,
                },
            },
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(l=15, r=15, t=35, b=10),
        height=205,
        font=dict(color="#E5E7EB"),
    )
    return fig

