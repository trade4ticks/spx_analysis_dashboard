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
    from reportlab.lib.enums import TA_LEFT
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


def _styles():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ResTitle", parent=styles["Title"],
        fontSize=20, spaceAfter=6, textColor=colors.HexColor("#1a5276"),
    )
    h1 = ParagraphStyle(
        "ResH1", parent=styles["Heading1"],
        fontSize=14, spaceAfter=4, textColor=colors.HexColor("#1a5276"),
    )
    h2 = ParagraphStyle(
        "ResH2", parent=styles["Heading2"],
        fontSize=11, spaceAfter=3, textColor=colors.HexColor("#111111"),
    )
    body = ParagraphStyle(
        "ResBody", parent=styles["Normal"],
        fontSize=9, spaceAfter=4, textColor=colors.HexColor("#111111"),
        leading=14,
    )
    dim = ParagraphStyle(
        "ResDim", parent=styles["Normal"],
        fontSize=8, spaceAfter=2, textColor=colors.HexColor("#444444"),
    )
    tag = ParagraphStyle(
        "ResTag", parent=styles["Normal"],
        fontSize=7, spaceAfter=1, textColor=colors.HexColor("#1a5276"),
        leading=10,
    )
    return {"title": title_style, "h1": h1, "h2": h2, "body": body, "dim": dim, "tag": tag}


def _esc(text: str) -> str:
    """Escape XML special chars for reportlab Paragraph."""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def _png_image(png_bytes: bytes, max_width=6.5 * 72, max_height=4.5 * 72) -> Optional[Image]:
    if not png_bytes:
        return None
    buf = io.BytesIO(png_bytes)
    img = Image(buf)
    scale = min(max_width / img.imageWidth, max_height / img.imageHeight, 1.0)
    img.drawWidth  = img.imageWidth  * scale
    img.drawHeight = img.imageHeight * scale
    return img


def build_pdf_bytes(run: dict, results: list[dict], chart_rows: list[dict]) -> bytes:
    """Build PDF and return as bytes."""
    buf = io.BytesIO()
    cfg = run.get("config") or {}
    if isinstance(cfg, str):
        cfg = json.loads(cfg)

    if cfg.get("engine") == "v2":
        _build_v2(run, results, cfg, buf)
    else:
        _build_v1(run, results, chart_rows, buf)
    return buf.getvalue()


# ── V2 PDF (structured sections + workflow plan) ─────────────────────────────

def _build_v2(run: dict, results: list[dict], cfg: dict, dest):
    _check_reportlab()
    S = _styles()

    doc = SimpleDocTemplate(
        dest, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []

    # ── Title ─────────────────────────────────────────────────────────────
    story.append(Paragraph(_esc(run.get("name", "Research Run")), S["title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3498db")))
    story.append(Spacer(1, 6))

    created = run.get("created_at")
    if created:
        story.append(Paragraph(str(created)[:19], S["dim"]))

    # ── Research Question ─────────────────────────────────────────────────
    story.append(Spacer(1, 8))
    story.append(Paragraph("Research Question", S["h1"]))
    story.append(Paragraph(f"<i>{_esc(run.get('question', ''))}</i>", S["body"]))
    story.append(Spacer(1, 8))

    # ── Configuration ─────────────────────────────────────────────────────
    config_lines = []
    if cfg.get("table"):
        config_lines.append(f"Table: {cfg['table']}")
    if cfg.get("tickers"):
        config_lines.append(f"Tickers: {', '.join(cfg['tickers'])}")
    if cfg.get("date_from") or cfg.get("date_to"):
        config_lines.append(f"Date range: {cfg.get('date_from', 'earliest')} to {cfg.get('date_to', 'latest')}")
    if cfg.get("model"):
        config_lines.append(f"Model: {cfg['model']}")
    for ln in config_lines:
        story.append(Paragraph(_esc(ln), S["dim"]))

    # ── Workflow Plan ─────────────────────────────────────────────────────
    plan = cfg.get("workflow_plan") or {}
    if plan:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Workflow Plan", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 4))

        if plan.get("task_type"):
            story.append(Paragraph(f"<b>Task Type:</b> {_esc(plan['task_type'])}", S["body"]))
        if plan.get("depth"):
            story.append(Paragraph(f"<b>Depth:</b> {_esc(plan['depth'])}", S["body"]))
        if plan.get("task_reasoning"):
            story.append(Paragraph(f"<b>Reasoning:</b> {_esc(plan['task_reasoning'])}", S["body"]))

        if plan.get("hypotheses"):
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Hypotheses:</b>", S["body"]))
            for h in plan["hypotheses"]:
                story.append(Paragraph(f"&bull; {_esc(h)}", S["body"]))

        if plan.get("feature_columns"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(
                f"<b>Feature columns (X):</b> {_esc(', '.join(plan['feature_columns']))}", S["body"]))
        if plan.get("outcome_columns"):
            story.append(Paragraph(
                f"<b>Outcome columns (Y):</b> {_esc(', '.join(plan['outcome_columns']))}", S["body"]))

        if plan.get("key_questions"):
            story.append(Spacer(1, 4))
            story.append(Paragraph("<b>Key Questions:</b>", S["body"]))
            for q in plan["key_questions"]:
                story.append(Paragraph(f"&bull; {_esc(q)}", S["body"]))

        if plan.get("scan_focus"):
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"<b>Analysis Focus:</b> {_esc(plan['scan_focus'])}", S["body"]))

    # ── Report (from structured sections) ─────────────────────────────────
    summary = run.get("ai_summary") or ""
    sections = []
    try:
        parsed = json.loads(summary)
        if isinstance(parsed, list):
            sections = parsed
    except (json.JSONDecodeError, TypeError):
        # Legacy plain-text: wrap as single markdown section
        if summary and not summary.startswith("[RUNNING]"):
            sections = [{"type": "markdown", "content": summary}]

    if sections:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Research Report", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))

        for sec in sections:
            if sec.get("type") == "markdown":
                content = sec.get("content", "")
                _render_markdown_to_story(content, story, S)

            elif sec.get("type") == "chart":
                # Render chart as a data table since we can't render Chart.js server-side
                title = sec.get("title", "Chart")
                config = sec.get("config") or {}
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>{_esc(title)}</b>", S["h2"]))
                _render_chart_as_table(config, story, S)
                story.append(Spacer(1, 6))

    # ── Scan Results Summary Table ────────────────────────────────────────
    scans = [r for r in results if r.get("analysis_type") == "scan" and not (r.get("result") or {}).get("error")]
    if scans:
        scans.sort(key=lambda r: (r.get("result") or {}).get("composite_score", 0), reverse=True)
        story.append(PageBreak())
        story.append(Paragraph("Scan Results", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))

        header = ["Score", "Ticker", "Feature", "Outcome", "Pattern", "Consistency", "N"]
        rows = [header]
        for r in scans[:30]:
            rd = r.get("result") or {}
            rob = rd.get("robustness") or {}
            rows.append([
                f"{rd.get('composite_score', 0):.0f}",
                r.get("ticker") or "all",
                r.get("x_col", "")[:25],
                r.get("y_col", "")[:20],
                rd.get("pattern", "")[:15],
                f"{rob.get('yearly_consistency_pct', '?')}%",
                str(rd.get("n", "")),
            ])
        t = Table(rows, colWidths=[35, 40, 130, 90, 70, 55, 35])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111111")),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.HexColor("#ffffff"), colors.HexColor("#f4f7fb")]),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (5, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t)

    doc.build(story)


def _render_markdown_to_story(md: str, story: list, S: dict):
    """Convert simple markdown to reportlab Paragraphs."""
    if not md:
        return
    for line in md.split("\n"):
        stripped = line.strip()
        if not stripped:
            story.append(Spacer(1, 4))
            continue
        if stripped.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
            continue
        if stripped.startswith("### "):
            story.append(Paragraph(_esc(stripped[4:]), S["h2"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(_esc(stripped[3:]), S["h1"]))
        elif stripped.startswith("# "):
            story.append(Paragraph(_esc(stripped[2:]), S["h1"]))
        elif stripped.startswith("- ") or stripped.startswith("* "):
            # Handle **bold** in bullet text
            text = stripped[2:]
            text = _md_inline(text)
            story.append(Paragraph(f"&bull; {text}", S["body"]))
        else:
            text = _md_inline(stripped)
            story.append(Paragraph(text, S["body"]))


def _md_inline(text: str) -> str:
    """Convert **bold** and *italic* markdown to reportlab XML tags."""
    import re
    text = _esc(text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    # Inline code
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="8">\1</font>', text)
    return text


def _render_chart_as_table(config: dict, story: list, S: dict):
    """Render a Chart.js config as a reportlab data table."""
    data = config.get("data") or {}
    labels = data.get("labels") or []
    datasets = data.get("datasets") or []

    if not labels or not datasets:
        story.append(Paragraph("<i>(chart data not available for PDF)</i>", S["dim"]))
        return

    # Build table: first column is labels, then one column per dataset
    header = [""] + [ds.get("label", f"Series {i+1}") for i, ds in enumerate(datasets)]
    rows = [header]
    for i, label in enumerate(labels):
        row = [str(label)]
        for ds in datasets:
            vals = ds.get("data") or []
            if i < len(vals) and vals[i] is not None:
                v = vals[i]
                row.append(f"{v:.3f}" if isinstance(v, float) else str(v))
            else:
                row.append("")
        rows.append(row)

    # Limit to 30 rows for readability
    if len(rows) > 31:
        rows = rows[:31]
        rows.append(["..."] + ["..." for _ in datasets])

    n_cols = len(header)
    col_w = min(80, int(450 / n_cols))
    col_widths = [col_w] * n_cols

    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111111")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#ffffff"), colors.HexColor("#f4f7fb")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)


# ── V1 PDF (legacy — unchanged) ─────────────────────────────────────────────

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
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#dce9f5")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.HexColor("#1a5276")),
        ("TEXTCOLOR",   (0, 1), (-1, -1), colors.HexColor("#111111")),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#ffffff"), colors.HexColor("#f4f7fb")]),
        ("GRID",        (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
    ]))
    return t


def _build_v1(run: dict, results: list[dict], chart_rows: list[dict], dest):
    """Legacy v1 PDF builder."""
    _check_reportlab()
    S = _styles()

    doc = SimpleDocTemplate(
        dest, pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []

    story.append(Paragraph(_esc(run.get("name", "Research Run")), S["title"]))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3498db")))
    story.append(Spacer(1, 6))

    created = run.get("created_at")
    if created:
        story.append(Paragraph(str(created)[:19], S["dim"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Research Question", S["h1"]))
    story.append(Paragraph(_esc(run.get("question", "")), S["body"]))
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
        story.append(Paragraph(_esc(ln), S["dim"]))

    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#444")))

    summary = run.get("ai_summary") or ""
    if summary:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Findings Summary", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 4))
        for para in summary.split("\n\n"):
            if para.strip():
                story.append(Paragraph(_esc(para.strip().replace("\n", " ")), S["body"]))
                story.append(Spacer(1, 4))

    if chart_rows:
        story.append(PageBreak())
        story.append(Paragraph("Charts & Analysis", S["h1"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#3498db")))
        story.append(Spacer(1, 6))
        for ch in chart_rows:
            png = ch.get("png_data")
            title = ch.get("title") or ""
            if png:
                story.append(Paragraph(_esc(title), S["h2"]))
                img = _png_image(bytes(png))
                if img:
                    story.append(img)
                story.append(Spacer(1, 10))

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
            label = f"{tk}  |  {x} &rarr; {y}"
            if spread is not None:
                label += f"   (D10-D1 spread: {spread*100:.3f}%)"
            story.append(Paragraph(label, S["h2"]))
            tbl = _decile_table(result_data, S)
            if tbl:
                story.append(tbl)
            story.append(Spacer(1, 8))

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
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce9f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#111111")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.HexColor("#ffffff"), colors.HexColor("#f4f7fb")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#aaaaaa")),
                ("ALIGN", (3, 0), (-1, -1), "CENTER"),
            ]))
            story.append(t)

    doc.build(story)
