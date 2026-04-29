"""Matplotlib chart generation for research results. Returns PNG bytes."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional

# ── Theme ─────────────────────────────────────────────────────────────────────
_BG      = "#1e1e1e"
_SURFACE = "#2d2d2d"
_TEXT    = "#cccccc"
_DIM     = "#888888"
_ACCENT  = "#3498db"
_POS     = "#3498db"   # bright blue for positive / top / long
_NEG     = "#e84393"   # bright pink for negative / bottom / short
_YELLOW  = "#f1c40f"


def _base_fig(figsize=(13, 6), title: str = ""):
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors=_TEXT, labelsize=11)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=13, pad=10)
    return fig, ax


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _label(ticker, feature, outcome=""):
    parts = [p for p in [ticker, feature, outcome] if p]
    return " | ".join(parts)


# ── Chart functions ────────────────────────────────────────────────────────────

def decile_bar_chart(result: dict) -> bytes:
    deciles = result.get("deciles", [])
    if not deciles:
        return b""
    xs = [d["decile"] for d in deciles]
    ys = [d["avg_ret"] or 0 for d in deciles]
    colors = [_POS if v >= 0 else _NEG for v in ys]

    title = "Decile Avg Return — " + _label(
        result.get("ticker", ""), result.get("feature_col", ""), result.get("outcome_col", "")
    )
    fig, ax = _base_fig(title=title)
    bars = ax.bar(xs, [v * 100 for v in ys], color=colors, alpha=0.85,
                  edgecolor="#111", linewidth=0.5)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xlabel("Decile  (1 = lowest feature value)", color=_TEXT)
    ax.set_ylabel("Avg Return (%)", color=_TEXT)
    ax.set_xticks(xs)

    for bar, val in zip(bars, ys):
        offset = 0.0002 if val >= 0 else -0.0002
        ax.text(bar.get_x() + bar.get_width() / 2,
                val * 100 + offset * 100,
                f"{val*100:.2f}%",
                ha="center", va="bottom" if val >= 0 else "top",
                color=_TEXT, fontsize=9)

    spread = result.get("top_bottom_spread")
    if spread is not None:
        ax.text(0.02, 0.97, f"D10–D1 spread: {spread*100:.2f}%",
                transform=ax.transAxes, color=_DIM, fontsize=10, va="top")
    return _to_png(fig)


_WHICH_LABEL = {
    "top":     "Top Decile (D10)",
    "top2":    "Top 2 Deciles (D9-D10)",
    "bottom":  "Bottom Decile (D1)",
    "bottom2": "Bottom 2 Deciles (D1-D2)",
}


def equity_curve_chart(top: dict, bottom: Optional[dict] = None) -> bytes:
    title = "Equity Curve — " + _label(
        top.get("ticker", ""), top.get("feature_col", ""), top.get("outcome_col", "")
    )
    fig, ax = _base_fig(figsize=(12, 5), title=title)

    def _plot(result, fallback_label, color):
        pts = result.get("points", [])
        if not pts:
            return
        label = _WHICH_LABEL.get(result.get("which"), fallback_label)
        idxs = list(range(len(pts)))
        vals = [p["value"] for p in pts]
        ax.plot(idxs, vals, color=color, linewidth=1.2, label=label)
        step = max(1, len(pts) // 8)
        tick_idxs = idxs[::step]
        tick_labels = [pts[i]["date"][:7] for i in tick_idxs]
        ax.set_xticks(tick_idxs)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)

    _plot(top, "Top", _POS)
    if bottom:
        _plot(bottom, "Bottom", _NEG)

    ax.axhline(0.0, color="#555", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Cumulative Return", color=_TEXT)
    ax.legend(facecolor=_SURFACE, edgecolor="#444", labelcolor=_TEXT, fontsize=10)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    # Show stats for both top and bottom
    anno_y = 0.04
    for curve, label, color in [(top, "Top", _POS), (bottom, "Bot", _NEG)]:
        if curve is None:
            continue
        n = curve.get("n_trades")
        cum = curve.get("cumulative_return") or curve.get("final_equity")
        dd = curve.get("max_drawdown")
        wr = curve.get("win_rate")
        if n is not None and cum is not None:
            info = (f"{label}: n={n}  Cum={cum*100:.1f}%  "
                    f"MaxDD={dd*100:.1f}%  WR={wr*100:.0f}%")
            ax.text(0.02, anno_y, info, transform=ax.transAxes,
                    color=color, fontsize=9)
            anno_y += 0.06
    note = top.get("concentration_note")
    if note:
        ax.text(0.02, anno_y, f"⚠ {note}",
                transform=ax.transAxes, color=_YELLOW, fontsize=8, va="bottom")
    return _to_png(fig)


def yearly_consistency_chart(result: dict) -> bytes:
    years = result.get("years", [])
    if not years:
        return b""
    title = "Yearly Consistency — " + _label(
        result.get("ticker", ""), result.get("feature_col", ""), result.get("outcome_col", "")
    )
    fig, ax = _base_fig(figsize=(12, 5), title=title)

    labels = [str(y["year"]) for y in years]
    top_vals = [(y["top_avg"] or 0) * 100 for y in years]
    bot_vals = [(y["bot_avg"] or 0) * 100 for y in years]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, top_vals, w, label="Top Decile", color=_POS, alpha=0.8)
    ax.bar(x + w / 2, bot_vals, w, label="Bottom Decile", color=_NEG, alpha=0.8)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=10)
    ax.set_ylabel("Avg Return (%)", color=_TEXT)
    ax.legend(facecolor=_SURFACE, edgecolor="#444", labelcolor=_TEXT, fontsize=10)

    pct = result.get("consistency_pct")
    if pct is not None:
        ax.text(0.02, 0.97, f"Top beats bottom: {pct:.0f}% of years",
                transform=ax.transAxes, color=_DIM, fontsize=10, va="top")
    return _to_png(fig)


def scatter_chart(result: dict) -> bytes:
    pts = result.get("points", [])
    if not pts:
        return b""
    title = "Scatter — " + _label(
        result.get("ticker", ""), result.get("x_col", ""), result.get("y_col", "")
    )
    xs = [p["x"] for p in pts]
    ys = [p["y"] for p in pts]

    fig, ax = _base_fig(title=title)
    ax.scatter(xs, ys, s=5, alpha=0.45, color=_ACCENT, edgecolors="none")

    # Regression overlay
    m, b = np.polyfit(xs, ys, 1)
    xl = np.linspace(min(xs), max(xs), 100)
    ax.plot(xl, m * xl + b, color=_YELLOW, linewidth=1.2, alpha=0.7)

    ax.set_xlabel(result.get("x_col", ""), color=_TEXT)
    ax.set_ylabel(result.get("y_col", ""), color=_TEXT)
    ax.text(0.02, 0.97, f"n={len(pts)}", transform=ax.transAxes,
            color=_DIM, fontsize=10, va="top")
    return _to_png(fig)


def correlation_heatmap(
    corr_results: list[dict],
    tickers: list[str],
    x_cols: list[str],
    y_cols: list[str],
) -> bytes:
    """One heatmap showing Pearson r for every ticker × (x→y) combo."""
    combos = [(x, y) for x in x_cols for y in y_cols]
    combo_labels = [f"{x}\n→{y}" for x, y in combos]

    data = np.full((len(tickers), len(combos)), np.nan)
    r_map = {
        (r.get("ticker"), r.get("x_col"), r.get("y_col")): r.get("pearson_r")
        for r in corr_results
        if "pearson_r" in r
    }
    for i, tk in enumerate(tickers):
        for j, (x, y) in enumerate(combos):
            val = r_map.get((tk, x, y))
            if val is not None:
                data[i, j] = val

    w = max(8, len(combos) * 1.8)
    h = max(3, len(tickers) * 0.65)
    fig, ax = plt.subplots(figsize=(w, h), facecolor=_BG)
    ax.set_facecolor(_BG)

    im = ax.imshow(data, cmap="RdYlGn", vmin=-0.3, vmax=0.3, aspect="auto")
    for i in range(len(tickers)):
        for j in range(len(combos)):
            val = data[i, j]
            if not np.isnan(val):
                fc = "black" if abs(val) < 0.15 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=fc, fontsize=8)

    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combo_labels, rotation=0, fontsize=9, color=_TEXT)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=11, color=_TEXT)
    ax.set_title("Pearson Correlation Heatmap", color=_TEXT, fontsize=13, pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=_TEXT, labelsize=10)
    cbar.set_label("Pearson r", color=_TEXT, fontsize=10)

    plt.tight_layout()
    return _to_png(fig)


def bucket_profile_chart(scan: dict) -> bytes:
    """Multi-metric bucket chart from scanner.scan_relationship output.
    Shows avg return bars + win rate line + highlights best zone."""
    bucket_stats = scan.get("bucket_stats") or []
    valid = [b for b in bucket_stats if b is not None]
    if len(valid) < 3:
        return b""

    xs = [b["bucket"] for b in valid]
    avgs = [b["avg_ret"] * 100 for b in valid]
    win_rates = [b["win_rate"] * 100 for b in valid]

    title = "Bucket Profile — " + _label(
        scan.get("ticker", ""), scan.get("x_col", ""), scan.get("y_col", ""))

    fig, ax1 = plt.subplots(figsize=(13, 6), facecolor=_BG)
    ax1.set_facecolor(_SURFACE)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.tick_params(colors=_TEXT, labelsize=11)

    # Bars: avg return
    colors = [_POS if v >= 0 else _NEG for v in avgs]
    bars = ax1.bar(xs, avgs, color=colors, alpha=0.80, edgecolor="#111", linewidth=0.5)
    ax1.axhline(0, color="#555", linewidth=0.8)
    ax1.set_xlabel("Bucket  (1 = lowest feature value)", color=_TEXT)
    ax1.set_ylabel("Avg Return (%)", color=_TEXT)
    ax1.set_xticks(xs)
    ax1.set_title(title, color=_TEXT, fontsize=13, pad=10)

    # Win rate line on secondary axis — dynamic range with padding
    ax2 = ax1.twinx()
    ax2.plot(xs, win_rates, color=_YELLOW, linewidth=1.5, marker="o", markersize=4)
    ax2.set_ylabel("Win Rate (%)", color=_YELLOW)
    ax2.tick_params(axis="y", colors=_YELLOW, labelsize=10)
    wr_min = min(win_rates) if win_rates else 40
    wr_max = max(win_rates) if win_rates else 60
    wr_pad = max((wr_max - wr_min) * 0.3, 5)
    ax2.set_ylim(max(0, wr_min - wr_pad), min(100, wr_max + wr_pad))

    # Highlight best adjacent zone
    bz = scan.get("best_adjacent_zone")
    if bz and bz.get("buckets"):
        for b_num in bz["buckets"]:
            ax1.axvspan(b_num - 0.4, b_num + 0.4, alpha=0.12, color=_ACCENT)

    # Annotations
    anno = []
    anno.append(f"Pattern: {scan.get('pattern', '?')}")
    anno.append(f"Score: {scan.get('composite_score', 0):.0f}/100")
    anno.append(f"Pearson: {scan.get('pearson_r', 0):.4f}  Spearman: {scan.get('spearman_r', 0):.4f}")
    rob = scan.get("robustness", {})
    if rob.get("yearly_consistency_pct") is not None:
        anno.append(f"Yearly consistency: {rob['yearly_consistency_pct']}%")
    if rob.get("half_sample_stable") is not None:
        anno.append(f"Half-sample stable: {'Yes' if rob['half_sample_stable'] else 'No'}")
    ax1.text(0.02, 0.97, "\n".join(anno), transform=ax1.transAxes,
             color=_DIM, fontsize=9, va="top", family="monospace")

    plt.tight_layout()
    return _to_png(fig)


def quadrant_chart(interaction: dict) -> bytes:
    """Bar chart showing avg return per quadrant from a 2-factor interaction scan."""
    quads = interaction.get("quadrants") or []
    valid = [q for q in quads if q.get("n", 0) >= 5 and q.get("avg_ret") is not None]
    if len(valid) < 2:
        return b""

    combo = interaction.get("combo", [])
    title = "2-Factor Interaction — " + _label(
        interaction.get("ticker", ""),
        " + ".join(combo),
        interaction.get("y_col", ""))

    labels = [q["label"] for q in valid]
    avgs = [q["avg_ret"] * 100 for q in valid]
    win_rates = [q.get("win_rate", 0.5) * 100 for q in valid]
    colors = [_POS if v >= 0 else _NEG for v in avgs]

    fig, ax1 = plt.subplots(figsize=(10, 5), facecolor=_BG)
    ax1.set_facecolor(_SURFACE)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")
    ax1.tick_params(colors=_TEXT, labelsize=10)

    bars = ax1.bar(range(len(valid)), avgs, color=colors, alpha=0.85,
                   edgecolor="#111", linewidth=0.5)
    ax1.axhline(0, color="#555", linewidth=0.8)
    ax1.set_xticks(range(len(valid)))
    ax1.set_xticklabels(labels, fontsize=8, color=_TEXT, rotation=15, ha="right")
    ax1.set_ylabel("Avg Return (%)", color=_TEXT)
    ax1.set_title(title, color=_TEXT, fontsize=12, pad=10)

    for bar, val, wr in zip(bars, avgs, win_rates):
        y_off = 0.003 if val >= 0 else -0.003
        ax1.text(bar.get_x() + bar.get_width() / 2, val + y_off,
                 f"{val:.2f}%\nWR:{wr:.0f}%",
                 ha="center", va="bottom" if val >= 0 else "top",
                 color=_TEXT, fontsize=8)

    # Annotations
    lift = interaction.get("interaction_lift", 0)
    r2 = interaction.get("ols_r2", 0)
    score = interaction.get("composite_interaction_score", 0)
    ax1.text(0.02, 0.97,
             f"Lift: {lift:+.4f}  R²: {r2:.4f}  Score: {score:.0f}/100\n"
             f"n={interaction.get('n', 0)}",
             transform=ax1.transAxes, color=_DIM, fontsize=9, va="top", family="monospace")

    plt.tight_layout()
    return _to_png(fig)
