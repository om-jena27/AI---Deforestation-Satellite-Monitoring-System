"""
utils/report.py
===============
PDF report generation using ReportLab.

generate_pdf_report(results, images) → bytes
"""

from __future__ import annotations

import io
import datetime
from typing import Any

import numpy as np

# ReportLab imports (installed: reportlab>=4)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
DARK_BG   = colors.HexColor("#0e1117")
GREEN     = colors.HexColor("#2ecc71")
RED       = colors.HexColor("#e74c3c")
ORANGE    = colors.HexColor("#e67e22")
YELLOW    = colors.HexColor("#f1c40f")
GREY_DARK = colors.HexColor("#1e2530")
GREY_MID  = colors.HexColor("#4a5568")
WHITE     = colors.white
ACCENT    = colors.HexColor("#00b4d8")


# ---------------------------------------------------------------------------
# Helper: convert numpy array → ReportLab Image flowable
# ---------------------------------------------------------------------------
def _np_to_rl_image(arr: np.ndarray, width_cm: float = 8, height_cm: float | None = None) -> RLImage:
    from PIL import Image as PILImage
    buf = io.BytesIO()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    PILImage.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    rl_img = RLImage(buf, width=width_cm * cm)
    if height_cm is not None:
        rl_img._height = height_cm * cm
    return rl_img


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------
def generate_pdf_report(
    results: dict[str, Any],
    images: dict[str, np.ndarray | None] | None = None,
) -> bytes:
    """
    Generate a professional PDF deforestation analysis report.

    Parameters
    ----------
    results : dict — output from calculate_area_metrics + change detection stats
              Expected keys (all optional — handled gracefully):
                forest_loss_pct, forest_t1_pixels, forest_t2_pixels,
                deforest_pixels, forest_t1_ha, forest_t2_ha, deforest_ha,
                forest_t1_km2, deforest_km2, co2_loss_tonnes,
                risk_label, risk_level, pixel_res_m, geo_available,
                forest_retained_pct, new_veg_pct,
                mean_confidence, max_confidence, min_confidence,
                mode_label, is_trained, note
    images : dict — optional numpy arrays (H, W, 3) uint8:
                t1_img, t2_img, deforest_overlay, deforest_mask_display,
                forest_mask_t1, forest_mask_t2, confidence_heatmap_arr

    Returns
    -------
    bytes — PDF file content
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="AI Deforestation Monitoring Report",
        author="AI Deforestation Monitoring System",
    )

    styles = getSampleStyleSheet()
    # Custom styles
    h1 = ParagraphStyle("H1", parent=styles["Title"],
                         fontSize=22, textColor=WHITE, backColor=DARK_BG,
                         spaceAfter=8, alignment=TA_CENTER)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"],
                         fontSize=14, textColor=ACCENT, spaceBefore=14, spaceAfter=4)
    body = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=10, textColor=colors.HexColor("#cfd8dc"),
                          spaceAfter=4, leading=14)
    note_style = ParagraphStyle("Note", parent=body,
                                fontSize=9, textColor=GREY_MID, leftIndent=12)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    story: list = []

    # ------------------------------------------------------------------ #
    # COVER
    # ------------------------------------------------------------------ #
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph("🌲 AI Deforestation Monitoring System", h1))
    story.append(Paragraph("Automated Analysis Report", h1))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated: {now}", body))

    mode_label = results.get("mode_label", "DEMO — Vegetation Index Proxy")
    is_trained = results.get("is_trained", False)
    mode_colour = GREEN if is_trained else ORANGE
    story.append(
        Paragraph(
            f'<font color="#{mode_colour.hexval()[2:]}">'
            f'Inference mode: {mode_label}</font>',
            body,
        )
    )

    story.append(Spacer(1, 1 * cm))

    # ------------------------------------------------------------------ #
    # EXECUTIVE SUMMARY
    # ------------------------------------------------------------------ #
    story.append(Paragraph("Executive Summary", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_MID))
    story.append(Spacer(1, 0.2 * cm))

    loss_pct = results.get("forest_loss_pct", 0.0)
    risk_label = results.get("risk_label", "N/A")

    summary_data = [
        ["Metric", "Value"],
        ["Forest Loss", f"{loss_pct:.2f} %"],
        ["Risk Level (project-defined)", risk_label],
        ["Forest (Time 1)", _fmt_area(results, "forest_t1")],
        ["Forest (Time 2)", _fmt_area(results, "forest_t2")],
        ["Deforested Area", _fmt_area(results, "deforest")],
        ["CO₂ Loss Estimate", _fmt_co2(results)],
        ["Pixel Resolution", _fmt_res(results)],
    ]

    t = Table(summary_data, colWidths=[9 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), GREY_DARK),
        ("TEXTCOLOR",  (0, 0), (-1, 0), ACCENT),
        ("FONTSIZE",   (0, 0), (-1, 0), 11),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#161b22")),
        ("TEXTCOLOR",  (0, 1), (-1, -1), colors.HexColor("#cfd8dc")),
        ("FONTSIZE",   (0, 1), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#161b22"), colors.HexColor("#1a2030")]),
        ("GRID", (0, 0), (-1, -1), 0.3, GREY_MID),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * cm))

    note = results.get("note", "")
    if note:
        story.append(Paragraph(f"⚠ {note}", note_style))

    # ------------------------------------------------------------------ #
    # IMAGE SECTION
    # ------------------------------------------------------------------ #
    if images:
        story.append(PageBreak())
        story.append(Paragraph("Satellite Image & AI Detection Visualizations", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_MID))
        story.append(Spacer(1, 0.3 * cm))

        img_pairs: list[tuple[str, str]] = [
            ("t1_img",                "Time 1 — Baseline Historical Satellite Image"),
            ("t2_img",                "Time 2 — Current Satellite Image"),
            ("forest_mask_t1",        "Forest Canopy Segmentation — Time 1 (Green = Forest)"),
            ("forest_mask_t2",        "Forest Canopy Segmentation — Time 2 (Green = Forest)"),
            ("deforest_overlay",      "AI Deforestation Overlay (Red = Detected Forest Loss)"),
            ("deforest_mask_display", "Binary Deforestation Mask (White = Loss, Black = Retained)"),
            ("confidence_heatmap",    "Model Confidence Heatmap (Probability of Detection)"),
        ]

        for key, caption in img_pairs:
            arr = images.get(key)
            if arr is not None:
                try:
                    rl_img = _np_to_rl_image(arr, width_cm=14)
                    story.append(Paragraph(caption, body))
                    story.append(rl_img)
                    story.append(Spacer(1, 0.4 * cm))
                except Exception:
                    story.append(Paragraph(f"[Image unavailable: {caption}]", note_style))

    # ------------------------------------------------------------------ #
    # CONFIDENCE SECTION
    # ------------------------------------------------------------------ #
    conf_mean = results.get("mean_confidence")
    conf_max  = results.get("max_confidence")
    conf_min  = results.get("min_confidence")

    if conf_mean is not None:
        story.append(Paragraph("Model Confidence & Statistical Quality", h2))
        story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_MID))
        story.append(Spacer(1, 0.2 * cm))

        conf_data = [
            ["Statistic", "Value"],
            ["Mean Confidence",    f"{conf_mean:.3f}  ({conf_mean*100:.1f}%)"],
            ["Max Confidence",     f"{conf_max:.3f}  ({conf_max*100:.1f}%)"],
            ["Min Confidence",     f"{conf_min:.3f}  ({conf_min*100:.1f}%)"],
            ["Inference Engine",   results.get("mode_label", "Deep Learning U-Net")],
        ]
        ct = Table(conf_data, colWidths=[9 * cm, 8 * cm])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), GREY_DARK),
            ("TEXTCOLOR",  (0, 0), (-1, 0), ACCENT),
            ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#161b22")),
            ("TEXTCOLOR",  (0, 1), (-1, -1), colors.HexColor("#cfd8dc")),
            ("GRID", (0, 0), (-1, -1), 0.3, GREY_MID),
            ("TOPPADDING",    (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        story.append(ct)
        story.append(Spacer(1, 0.3 * cm))

    # ------------------------------------------------------------------ #
    # MODEL INFORMATION & LIMITATIONS
    # ------------------------------------------------------------------ #
    story.append(Paragraph("Model Information & Limitations", h2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_MID))
    story.append(Spacer(1, 0.2 * cm))

    model_info_text = (
        "• <b>Architecture:</b> Deep Convolutional U-Net with symmetric encoder-decoder "
        "skip connections for pixel-level semantic forest canopy segmentation.<br/>"
        "• <b>Input Channels:</b> 3-band RGB / Multispectral optical reflectance.<br/>"
        "• <b>Loss Function:</b> Combined Binary Cross-Entropy + Soft Dice Loss for class-imbalanced terrain.<br/>"
        "• <b>Limitations:</b> Optical satellite imagery is susceptible to persistent cloud cover, atmospheric haze, "
        "and seasonal phenological leaf drop. Synthetic aperture radar (SAR) or multi-temporal compositing is recommended for cloud-affected scenes."
    )
    story.append(Paragraph(model_info_text, body))
    story.append(Spacer(1, 0.5 * cm))

    # ------------------------------------------------------------------ #
    # DISCLAIMER (MANDATORY EXACT TEXT)
    # ------------------------------------------------------------------ #
    story.append(Spacer(1, 0.5 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=GREY_MID))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        "<b>Important Environmental & Legal Notice:</b><br/>"
        "This system is an AI-assisted analysis tool and results should be verified using "
        "authoritative remote-sensing or field data before environmental/legal decisions.",
        note_style,
    ))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    return pdf_bytes



# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_area(r: dict, prefix: str) -> str:
    ha  = r.get(f"{prefix}_ha")
    km2 = r.get(f"{prefix}_km2")
    px  = r.get(f"{prefix}_pixels")
    if ha is not None:
        return f"{ha:,.2f} ha  ({km2:.4f} km²)"
    if px is not None:
        return f"{px:,} px  (resolution unknown)"
    return "N/A"


def _fmt_co2(r: dict) -> str:
    co2 = r.get("co2_loss_tonnes")
    if co2 is not None:
        return f"~{co2:,.1f} tonnes CO₂"
    return "Requires pixel resolution"


def _fmt_res(r: dict) -> str:
    res = r.get("pixel_res_m")
    if res is not None:
        return f"{res:.1f} m/pixel"
    return "Not specified"
