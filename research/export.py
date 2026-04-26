"""PDF report generation from a completed research run."""
import io
import json
from datetime import datetime
from typing import Optional

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image,
        Table, TableStyle, PageBreak, HRFlowable,
    )
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False


def _check_reportlab():
    if not _REPORTLAB:
        raise ImportError("reportlab is not installed. Run: pip install reportlab")


def _dark_style():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ResTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#3498db"),
    )
    h1 = ParagraphStyle(
        "ResH1", parent=styles["Heading1"],
        fontSize=14, spaceAfter=4, textColor=colors.HexColor("#3498db"),
    )
    h2 = ParagraphStyle(
        "ResH2", parent=styles["Heading2"],
        fontSize=11, spaceAfter=3, textColor=colors.HexColor("#cccccc"),
    )
    body = ParagraphStyle(
        "ResBody", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, textColor=colors.HexColor("#cccccc"),
        leading=13,
    )
    dim = ParagraphStyle(
        "ResDim", parent=styles["Normal"],
        fontSize=8, spaceAfter=2, textColor=colors.HexColor("#888888"),
    )
    return {"title": title_style, "h1": h1, "h2": h2, "body": body, "dim": dim}


def _png_image(png_bytes: bytes, max_width=6.5 * 72, max_height=4 * 72) -> Optional[Image]:
    if not png_bytes:
        return None
    buf = io.BytesIO(png_bytes)
    img = Image(buf)
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    return img


def _decile_table(decile_result: dict, styles):
    deciles = decile_result.get("deciles", [])
    if not deciles:
        return None
    header = ["Decile", "N", "Avg Ret %", "Median %", "Win Rate %", "Std Dev %"]
    rows = [header]
    for d in deciles:
        rows.append([
            str(d["decile"]),
            str(d["n"]),
            f"{(d['avg_ret'] or 0)*100:.3f}",
            f"{(d['med_ret'] or 0)*100:.3f}",
            f"{(d['win_rate'] or 0)*100:.1f}",
            f"{(d['std_dev'] or 0)*100:.3f}",
        ])
    t = Table(rows, colWidths=[50, 45, 65, 65, 70, 65])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2d2d2d")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.HexColor("#3498db")),
        ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#cccccc")),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#1e1e1e"), colors.HexColor("#262626")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#444")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def build_pdf_bytes(run: dict, results: list[dict], chart_rows: list[dict]) -> bytes:
    """Build PDF and return as bytes (for HTTP download)."""
    buf = io.BytesIO()
    _build(run, results, chart_rows, buf)
    return buf.getvalue()


def build_pdf(run: dict, results: list[dict], chart_rows: list[dict],
              output_path: str):
    """Assemble and write the PDF report to output_path (for CLI use)."""
    with open(output_path, "wb") as f:
        f.write(build_pdf_bytes(run, results, chart_rows))


def _build(run: dict, results: list[dict], chart_rows: list[dict], dest):
    """Internal: write PDF to dest (file path str or file-like object)."""
    _check_reportlab()
    S = _dark_style()

    doc = SimpleDocTemplate(
        dest,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    story = []

    # ── Title page ────────────────────────────────────────────────────────────
    story.append(Paragraph(run.get("name", "Research Run"), S["title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3498db")))
    story.append(Spacer(1, 6))

    created = run.get("created_at")
    if created:
        story.append(Paragraph(str(created)[:19], S["dim"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Research Question", S["h1"]))
    story.append(Paragraph(run.get("question", ""), S["body"]))
    story.append(Spacer(1, 8))

    cfg = run.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    lines = []
    if cfg.get("tickers"):
        lines.append(f"Tickers: {', '.join(cfg['tickers'])}")
    if cfg.get("x_columns"):
        lines.append(f"Features: {', '.join(cfg['x_columns'])}")
    if cfg.get("y_columns"):
        lines.append(f"Outcomes: {', '.join(cfg['y_columns'])}")
    if cfg.get("table"):
        lines.append(f"Table: {cfg['table']}")
    for ln in lines:
        story.append(Paragraph(ln, S["dim"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#444")))

    # ── AI Summary ────────────────────────────────────────────────────────────
    summary = run.get("ai_summary") or ""
    if summary:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Findings Summary", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 4))
        for para in summary.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip().replace("\n", " "), S["body"]))
                story.append(Spacer(1, 4))

    # ── Charts ────────────────────────────────────────────────────────────────
    if chart_rows:
        story.append(PageBreak())
        story.append(Paragraph("Charts & Analysis", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))

        for ch in chart_rows:
            png = ch.get("png_data")
            title = ch.get("title") or ""
            if png:
                story.append(Paragraph(title, S["h2"]))
                img = _png_image(bytes(png))
                if img:
                    story.append(img)
                story.append(Spacer(1, 10))

    # ── Decile stats tables ───────────────────────────────────────────────────
    decile_results = [r for r in results if r.get("analysis_type") == "decile"]
    if decile_results:
        story.append(PageBreak())
        story.append(Paragraph("Decile Statistics", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))

        for r in decile_results:
            result_data = r.get("result") or {}
            if isinstance(result_data, str):
                result_data = json.loads(result_data)
            tk = r.get("ticker") or "all"
            x = r.get("x_col", "")
            y = r.get("y_col", "")
            spread = result_data.get("top_bottom_spread")
            label = f"{tk}  |  {x} → {y}"
            if spread is not None:
                label += f"   (D10–D1 spread: {spread*100:.3f}%)"
            story.append(Paragraph(label, S["h2"]))
            tbl = _decile_table(result_data, S)
            if tbl:
                story.append(tbl)
            story.append(Spacer(1, 8))

    # ── Correlation summary table ─────────────────────────────────────────────
    corr_results = [r for r in results if r.get("analysis_type") == "correlation"]
    if corr_results:
        story.append(PageBreak())
        story.append(Paragraph("Correlation Summary", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))

        header = ["Ticker", "Feature", "Outcome", "Pearson r", "p-value", "Spearman r", "N"]
        rows = [header]
        for r in corr_results:
            rd = r.get("result") or {}
            if isinstance(rd, str):
                rd = json.loads(rd)
            if "error" in rd:
                continue
            rows.append([
                r.get("ticker") or "all",
                r.get("x_col", ""),
                r.get("y_col", ""),
                f"{rd.get('pearson_r', ''):.4f}",
                f"{rd.get('pearson_p', ''):.4f}",
                f"{rd.get('spearman_r', ''):.4f}",
                str(rd.get("n", "")),
            ])
        if len(rows) > 1:
            t = Table(rows, colWidths=[45, 130, 85, 55, 55, 60, 40])
            t.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2d2d2d")),
                ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.HexColor("#3498db")),
                ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#cccccc")),
                ("FONTSIZE",    (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#1e1e1e"), colors.HexColor("#262626")]),
                ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#444")),
                ("ALIGN",       (3, 0), (-1, -1), "CENTER"),
            ]))
            story.append(t)

    doc.build(story)
