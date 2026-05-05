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
    from reportlab.graphics.shapes import Drawing, String, Line, Rect
    from reportlab.graphics import renderPDF
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


def _safe_hex(color_str, fallback):
    """Return a reportlab color from a hex string, falling back on rgba() or bad input."""
    if not isinstance(color_str, str):
        return fallback
    s = color_str.strip()
    if s.startswith("#"):
        try:
            return colors.HexColor(s)
        except Exception:
            return fallback
    return fallback


def _esc(text: str) -> str:
    """Escape XML special chars for reportlab Paragraph."""
    s = str(text)
    # Replace curly quotes and em-dashes with ASCII equivalents ReportLab can handle
    s = s.replace("‘", "'").replace("’", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "--")
    s = s.replace("•", "*").replace("…", "...")
    # XML escapes must come last
    return (s
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
    if cfg.get("intratrade_mode") and cfg.get("backtest_upload_id"):
        config_lines.append("Mode: Intratrade Path Analysis (agentic)")
    elif cfg.get("backtest_upload_id"):
        config_lines.append("Mode: Backtest Trade Regime Analysis (agentic)")
    elif cfg.get("pnl_upload_id"):
        config_lines.append("Mode: P&L–IV Correlation (agentic analysis)")
    elif cfg.get("table"):
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
                _render_chart_drawing(config, story, S)
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


def _render_chart_drawing(config: dict, story: list, S: dict):
    """Render a Chart.js config as a reportlab Drawing (bar or line chart)."""
    data = config.get("data") or {}
    labels = data.get("labels") or []
    datasets = data.get("datasets") or []
    chart_type = config.get("type", "bar")
    is_horizontal = (config.get("options") or {}).get("indexAxis") == "y"

    if not labels or not datasets:
        story.append(Paragraph("<i>(chart data not available)</i>", S["dim"]))
        return

    W, H = 460, 200
    margin_l, margin_b, margin_r, margin_t = 50, 30, 20, 15
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_b - margin_t

    d = Drawing(W, H)

    # Background
    d.add(Rect(0, 0, W, H, fillColor=colors.HexColor("#f8f9fa"), strokeColor=None))

    # Collect all values for axis scaling
    all_vals = []
    for ds in datasets:
        for v in (ds.get("data") or []):
            if v is not None and isinstance(v, (int, float)):
                all_vals.append(float(v))
    if not all_vals:
        story.append(Paragraph("<i>(no numeric data)</i>", S["dim"]))
        return

    v_min = min(0, min(all_vals))
    v_max = max(0, max(all_vals))
    v_range = v_max - v_min or 1

    ds_colors = [
        colors.HexColor("#3498db"), colors.HexColor("#e84393"),
        colors.HexColor("#2ecc71"), colors.HexColor("#e67e22"),
        colors.HexColor("#9b59b6"), colors.HexColor("#1abc9c"),
    ]

    if chart_type == "line":
        _draw_line_chart(d, datasets, labels, all_vals, v_min, v_max, v_range,
                         ds_colors, margin_l, margin_b, plot_w, plot_h)
    elif is_horizontal:
        _draw_horizontal_bar(d, datasets, labels, all_vals, v_min, v_max, v_range,
                             ds_colors, margin_l, margin_b, plot_w, plot_h, W, H)
    else:
        _draw_bar_chart(d, datasets, labels, all_vals, v_min, v_max, v_range,
                        ds_colors, margin_l, margin_b, plot_w, plot_h)

    # X-axis labels (for non-horizontal charts)
    if not is_horizontal:
        n = len(labels)
        step = max(1, n // 15)
        for i in range(0, n, step):
            x = margin_l + (i + 0.5) * plot_w / n
            d.add(String(x, margin_b - 12, str(labels[i])[:10],
                         fontSize=6, fillColor=colors.HexColor("#444444"),
                         textAnchor="middle"))

    # Y-axis labels
    if not is_horizontal:
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            val = v_min + frac * v_range
            y = margin_b + frac * plot_h
            d.add(String(margin_l - 4, y - 3, f"{val:.2f}",
                         fontSize=6, fillColor=colors.HexColor("#666666"),
                         textAnchor="end"))
            d.add(Line(margin_l, y, margin_l + plot_w, y,
                       strokeColor=colors.HexColor("#dddddd"), strokeWidth=0.3))

    # Zero line
    if v_min < 0 < v_max and not is_horizontal:
        zero_y = margin_b + (-v_min / v_range) * plot_h
        d.add(Line(margin_l, zero_y, margin_l + plot_w, zero_y,
                   strokeColor=colors.HexColor("#999999"), strokeWidth=0.5))

    # Legend
    x_legend = margin_l
    for i, ds in enumerate(datasets):
        label = ds.get("label", "")
        if not label:
            continue
        col = ds_colors[i % len(ds_colors)]
        # Use per-bar coloring color if single dataset
        if len(datasets) == 1 and isinstance(ds.get("backgroundColor"), list):
            col = colors.HexColor("#3498db")
        d.add(Rect(x_legend, H - 10, 8, 6, fillColor=col, strokeColor=None))
        d.add(String(x_legend + 10, H - 10, label[:30],
                     fontSize=6, fillColor=colors.HexColor("#444444")))
        x_legend += len(label) * 4.5 + 20

    story.append(d)


def _draw_bar_chart(d, datasets, labels, all_vals, v_min, v_max, v_range,
                    ds_colors, margin_l, margin_b, plot_w, plot_h):
    n = len(labels)
    n_ds = len(datasets)
    bar_group_w = plot_w / n
    bar_w = bar_group_w * 0.7 / max(n_ds, 1)
    gap = bar_group_w * 0.15

    for di, ds in enumerate(datasets):
        vals = ds.get("data") or []
        bg = ds.get("backgroundColor")
        for i in range(min(len(vals), n)):
            v = vals[i]
            if v is None:
                continue
            v = float(v)
            # Determine bar color
            fallback = ds_colors[di % len(ds_colors)]
            if isinstance(bg, list) and i < len(bg):
                col = _safe_hex(bg[i], fallback)
            elif isinstance(bg, str):
                col = _safe_hex(bg, fallback)
            else:
                col = fallback

            zero_y = margin_b + (-v_min / v_range) * plot_h
            bar_h = (v / v_range) * plot_h
            x = margin_l + gap + i * bar_group_w + di * bar_w
            if v >= 0:
                d.add(Rect(x, zero_y, bar_w, bar_h, fillColor=col, strokeColor=None))
            else:
                d.add(Rect(x, zero_y + bar_h, bar_w, -bar_h, fillColor=col, strokeColor=None))


def _draw_horizontal_bar(d, datasets, labels, all_vals, v_min, v_max, v_range,
                          ds_colors, margin_l, margin_b, plot_w, plot_h, W, H):
    n = len(labels)
    if not n:
        return
    # For horizontal bars, labels go on Y axis, values on X
    bar_h = plot_h / n * 0.7
    gap = plot_h / n * 0.15

    # Wider left margin for ticker labels
    h_margin_l = 60

    for di, ds in enumerate(datasets):
        vals = ds.get("data") or []
        bg = ds.get("backgroundColor")
        for i in range(min(len(vals), n)):
            v = vals[i]
            if v is None:
                continue
            v = float(v)
            fallback = ds_colors[di % len(ds_colors)]
            if isinstance(bg, list) and i < len(bg):
                col = _safe_hex(bg[i], fallback)
            elif isinstance(bg, str):
                col = _safe_hex(bg, fallback)
            else:
                col = fallback

            bar_w_px = (v / v_range) * (W - h_margin_l - 20)
            y = margin_b + (n - 1 - i) * (plot_h / n) + gap
            d.add(Rect(h_margin_l, y, max(bar_w_px, 1), bar_h, fillColor=col, strokeColor=None))
            # Value label
            d.add(String(h_margin_l + bar_w_px + 3, y + bar_h / 2 - 3, f"{v:.0f}",
                         fontSize=6, fillColor=colors.HexColor("#444444")))

    # Y-axis labels (ticker names)
    for i, label in enumerate(labels):
        y = margin_b + (n - 1 - i) * (plot_h / n) + gap + bar_h / 2 - 3
        d.add(String(h_margin_l - 4, y, str(label)[:12],
                     fontSize=6, fillColor=colors.HexColor("#444444"),
                     textAnchor="end"))


def _draw_line_chart(d, datasets, labels, all_vals, v_min, v_max, v_range,
                     ds_colors, margin_l, margin_b, plot_w, plot_h):
    n = len(labels)
    if n < 2:
        return

    for di, ds in enumerate(datasets):
        vals = ds.get("data") or []
        col = ds_colors[di % len(ds_colors)]
        col = _safe_hex(ds.get("borderColor"), col)

        prev_x, prev_y = None, None
        for i in range(min(len(vals), n)):
            v = vals[i]
            if v is None:
                prev_x, prev_y = None, None
                continue
            v = float(v)
            x = margin_l + i * plot_w / (n - 1)
            y = margin_b + ((v - v_min) / v_range) * plot_h
            if prev_x is not None:
                d.add(Line(prev_x, prev_y, x, y, strokeColor=col, strokeWidth=1.2))
            prev_x, prev_y = x, y


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
