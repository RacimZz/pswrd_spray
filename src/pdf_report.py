from __future__ import annotations
from pathlib import Path
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image

def generate_pdf(
    alerts: pd.DataFrame,
    metrics: dict,
    shap_summary_path: Path | None,
    out_path: Path,
) -> None:
    doc = SimpleDocTemplate(str(out_path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # ── Titre ──
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=6)
    story.append(Paragraph("Password Spray Detection — Rapport d'analyse", title_style))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Métriques clés ──
    story.append(Paragraph("Métriques clés", styles["Heading2"]))
    metric_data = [["Indicateur", "Valeur"]] + [[k, str(v)] for k, v in metrics.items()]
    t = Table(metric_data, colWidths=[9*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── Table des alertes ──
    story.append(Paragraph("Alertes détectées", styles["Heading2"]))
    cols = [c for c in ["priority", "src_ip", "ts", "anomaly_score",
                         "n_attempts", "n_fail", "fail_rate", "n_users"] if c in alerts.columns]
    top = alerts[cols].head(20).copy()
    if "anomaly_score" in top.columns:
        top["anomaly_score"] = top["anomaly_score"].round(4)
    if "fail_rate" in top.columns:
        top["fail_rate"] = top["fail_rate"].round(3)
    if "ts" in top.columns:
        top["ts"] = top["ts"].astype(str).str[:16]

    table_data = [cols] + top.values.tolist()
    col_w = [15*cm / len(cols)] * len(cols)
    t2 = Table(table_data, colWidths=col_w, repeatRows=1)
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",   (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdf2f2")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.5*cm))

    # ── SHAP summary (si dispo) ──
    if shap_summary_path and shap_summary_path.exists():
        story.append(Paragraph("Importance des features (SHAP)", styles["Heading2"]))
        story.append(Image(str(shap_summary_path), width=14*cm, height=8*cm))

    doc.build(story)
