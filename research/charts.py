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
_GREEN   = "#2ecc71"
_RED     = "#e74c3c"
_YELLOW  = "#f1c40f"


def _base_fig(figsize=(10, 5), title: str = ""):
    fig, ax = plt.subplots(figsize=figsize, facecolor=_BG)
    ax.set_facecolor(_SURFACE)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors=_TEXT, labelsize=8)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    if title:
        ax.set_title(title, color=_TEXT, fontsize=10, pad=8)
    return fig, ax


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
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
    colors = [_GREEN if v >= 0 else _RED for v in ys]

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
                color=_TEXT, fontsize=7)

    spread = result.get("top_bottom_spread")
    if spread is not None:
        ax.text(0.02, 0.97, f"D10–D1 spread: {spread*100:.2f}%",
                transform=ax.transAxes, color=_DIM, fontsize=8, va="top")
    return _to_png(fig)


def equity_curve_chart(top: dict, bottom: Optional[dict] = None) -> bytes:
    title = "Equity Curve — " + _label(
        top.get("ticker", ""), top.get("feature_col", ""), top.get("outcome_col", "")
    )
    fig, ax = _base_fig(figsize=(12, 5), title=title)

    def _plot(result, label, color):
        pts = result.get("points", [])
        if not pts:
            return
        idxs = list(range(len(pts)))
        vals = [p["value"] for p in pts]
        ax.plot(idxs, vals, color=color, linewidth=1.2, label=label)
        # Sparse x-axis date labels (every ~10% of trades)
        step = max(1, len(pts) // 8)
        tick_idxs = idxs[::step]
        tick_labels = [pts[i]["date"][:7] for i in tick_idxs]  # YYYY-MM
        ax.set_xticks(tick_idxs)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7)

    _plot(top, "Top Decile", _GREEN)
    if bottom:
        _plot(bottom, "Bottom Decile", _RED)

    ax.axhline(1.0, color="#555", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Equity (start = 1.0)", color=_TEXT)
    ax.legend(facecolor=_SURFACE, edgecolor="#444", labelcolor=_TEXT, fontsize=8)

    dd = top.get("max_drawdown")
    n = top.get("n_trades")
    final = top.get("final_equity")
    if n is not None:
        info = f"Trades: {n}  |  Final: {final:.2f}x  |  MaxDD: {dd*100:.1f}%"
        ax.text(0.02, 0.04, info, transform=ax.transAxes, color=_DIM, fontsize=8)
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
    ax.bar(x - w / 2, top_vals, w, label="Top Decile", color=_GREEN, alpha=0.8)
    ax.bar(x + w / 2, bot_vals, w, label="Bottom Decile", color=_RED, alpha=0.8)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, fontsize=8)
    ax.set_ylabel("Avg Return (%)", color=_TEXT)
    ax.legend(facecolor=_SURFACE, edgecolor="#444", labelcolor=_TEXT, fontsize=8)

    pct = result.get("consistency_pct")
    if pct is not None:
        ax.text(0.02, 0.97, f"Top beats bottom: {pct:.0f}% of years",
                transform=ax.transAxes, color=_DIM, fontsize=8, va="top")
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
    ax.plot(xl, m * xl + b, color=_RED, linewidth=1.2, alpha=0.7)

    ax.set_xlabel(result.get("x_col", ""), color=_TEXT)
    ax.set_ylabel(result.get("y_col", ""), color=_TEXT)
    ax.text(0.02, 0.97, f"n={len(pts)}", transform=ax.transAxes,
            color=_DIM, fontsize=8, va="top")
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
    ax.set_xticklabels(combo_labels, rotation=0, fontsize=7, color=_TEXT)
    ax.set_yticks(range(len(tickers)))
    ax.set_yticklabels(tickers, fontsize=9, color=_TEXT)
    ax.set_title("Pearson Correlation Heatmap", color=_TEXT, fontsize=11, pad=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=_TEXT, labelsize=8)
    cbar.set_label("Pearson r", color=_TEXT, fontsize=8)

    plt.tight_layout()
    return _to_png(fig)
