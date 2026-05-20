# SPX Dashboard — Session Handoff (2026-05-20)

This document is a self-contained briefing for picking up work on another machine. Read it top to bottom, then jump in.

## Project at a glance

- **Repo:** `C:\Personal\Data\spx_analysis_dashboard` (Windows machine; the new machine should clone the repo at an equivalent path or update absolute paths in commands below)
- **Stack:** FastAPI + Alpine.js + Chart.js + PostgreSQL
- **Deployment:** VPS named `Trading` at Tailscale `100.76.94.99:8000`, served from `/spx_analysis_dashboard` via `python run.py`. No systemd yet — process dies on SSH logout.
- **Branch:** `master`. All recent work is pushed.
- **GitHub remote:** `trade4ticks/spx_analysis_dashboard`

## The big arc — walk-forward bins for look-ahead-bias-free analysis

Every binning analysis on the OI Analysis page historically computed bin thresholds from the **full available history**. "Bin 20" meant top-5% in the entire dataset, not what would have been top-5% at the time of a historical trade — look-ahead bias.

We've been adding **walk-forward** mode: at each date D, bin assignment for the row at D uses only data from `[start, D)`. Per-ticker bisect_left against a running sorted list, with a **252-trading-day warmup** (~1 year of 2019 data) before bins are emitted.

The plan was structured as three phases (see `C:\Users\kroko\.claude\plans\stateful-tumbling-dragon.md` for the full plan).

### Phase status

| Phase | What | Status | Commit |
|------|------|--------|--------|
| **1** | Threshold Drift collapsable pane (visualize bin thresholds over time, median + IQR band per bin) | **DONE** | `cf7ed3f` through `2c89d1e` |
| **2** | Walk-forward toggle on Multi-Metric Correlation Explorer | **DONE** | `ab61c6f`, refined in `00236d8` |
| **3a** | Walk-forward primary `/analyze` endpoint + page-wide `pageMode` toggle (replaces corr-explorer-local toggle) | **DONE** | `951cc57` ← **just shipped** |
| **3b** | Walk-forward System Portfolio aggregate (`/portfolios/{pid}/aggregate`) | pending |
| **3c** | Walk-forward Secondary Signal Scanner | pending |
| **3d** | Walk-forward Score Matrix (heaviest — currently uses `_compute_all_bins_fast` vectorized) | pending |

## What the user wants next

Phase 3b / 3c / 3d. **Ask the user which they want first.** They've signed off on the page-wide architecture; the only open question is sequencing. My recommendation when they ask:

- **3b** is highest leverage (System Portfolio is the user's primary research tool right now).
- **3c** is straightforward (additive flag; reuse `_walk_forward_bins`).
- **3d** is the heavy lift — Score Matrix has 87 tickers × 80 metrics and currently runs near Cloudflare's 100s timeout in-sample. Walk-forward will be slower; may need its own numpy-vectorized walk-forward path or to land as an opt-in "Recompute walk-forward" button.

When 3b/c/d are all in, the per-section walk-forward UI in the corr explorer (already removed in 3a) stays gone — the global toggle is the single source of truth.

## ⚠️ Open issue (user is monitoring, not blocking)

The user reported that **some primary-section visuals stopped rendering** after the recent updates:
- Rolling Signal Strength (`chart-rolling`)
- Win Rate by Decile (`chart-winrate`)
- Return Distribution (`chart-dist`)
- Trade Data table
- Trade Calendar (`chart-calendar`)

These all sit at lines ~520–580 in `templates/oi_analysis.html`. Charts above this section (decile bar, equity, yearly, boxplot, drawdown) and below (DOW, activity, heatmap) reportedly DO render.

Nothing in the recent diffs obviously touches those render paths — all walk-forward work was in `/analyze`, the corr explorer, and additive helpers. The user said they'd "monitor and we can resolve later." If they bring it up:

1. Ask them to open DevTools (F12) → Console → hard refresh (Ctrl+Shift+R) → paste any red errors.
2. Confirm `oi_analysis.js?v=71` is fetched fresh (not 304 disk cache) in the Network tab.
3. Likely a cache or Alpine reactivity issue, not a real regression — but verify before assuming.

## Architecture cheat sheet

### Critical files

| File | Role |
|------|------|
| `app/routers/oi_analysis.py` | `/analyze`, `/threshold-drift`, `/secondary-corr-bins`, `/secondary-correlation`, `/global-metric-bins` endpoints. All walk-forward primitives live here. |
| `app/routers/oi_portfolios.py` | System Portfolio CRUD + `/portfolios/{pid}/aggregate` (Phase 3b will extend this). |
| `static/js/oi_analysis.js` | Main Alpine component (~3800 lines). Page state, all chart renderers, all fetch logic. |
| `templates/oi_analysis.html` | Page template. Cache version controlled by `<script src="/static/js/oi_analysis.js?v=NN">` line — bump on every JS change. Currently **v=71**. |
| `templates/oi_signals.html` + `static/js/oi_signals.js` | OI Signals page (signals integration; separate from OI Analysis). |

### Walk-forward primitives (in `oi_analysis.py`)

- `_DEFAULT_WALKFWD_WARMUP = 252` (constant; trading days)
- `_walk_forward_bins(rows_chrono, metric, n_bins, is_all, warmup)` — `{row_idx: bin_or_None}` for each row. Per-ticker in ALL mode.
- `_walk_forward_thresholds(...)` — time-series of per-ticker bin-K upper thresholds at canonical month-end. Used by Threshold Drift.
- `_walk_forward_bucket_pairs(pairs, n_bins_list, warmup)` — single-history multi-granularity (10-bin + 20-bin in one pass). Used by single-ticker `/analyze`.
- `_walk_forward_bucket_per_ticker(by_ticker, n_bins_list, warmup)` — ALL mode wrapper around the above. Used by ALL-mode `/analyze`.
- `_walk_forward_primary_filter(rows, primary_metric, selected_primary_bins, is_all)` → `(filtered_chrono, dropped, universe)`. Used by corr-explorer walk-forward branches.
- `_compute_walk_forward_bin_stats(...)` — per-feature walk-forward stats inside an already-filtered subset.
- `_walk_forward_membership(...)` — walk-forward analogue of `_bin_membership`.

All use `bisect_left` against a running sorted list. Bin formula: `min(int(rank / n_after * n_bins) + 1, n_bins)` — matches the in-sample `_bin_membership` exactly so the two modes are directly comparable.

### Frontend state (page-wide)

```js
// in static/js/oi_analysis.js
pageMode: 'in_sample',   // 'in_sample' | 'walk_forward' — replaces old corrMode
selectedBins20: new Set([1, 2, 19, 20]),   // 1..20 bin IDs of primary filter
corrBinCount: 20,
secCacheKey: null,
...
```

`setPageMode(m)` is the cascade trigger. Currently it re-runs `loadAnalysis()` and (if open) the corr explorer fetches. **Phase 3b/c/d will extend this cascade** — add calls to `loadPortfolioAggregate()`, `secScan()`, and the Score Matrix refresh.

### How walk_forward is plumbed

Every endpoint that takes the flag does it the same way:

1. **Backend Pydantic body / query param:** `walk_forward: bool = False` (plus `selected_primary_bins: Optional[List[int]] = None` where the primary filter is relevant).
2. **Backend branch:** `if walk_forward:` swaps in walk-forward primitives; in-sample branch untouched.
3. **Backend response:** add `mode`, `warmup`, `dropped_warmup_n`, `start_date` (and `universe_n` for corr endpoints where the primary filter narrows further).
4. **Frontend fetch:** `walk_forward: this.pageMode === 'walk_forward'`.
5. **Frontend UI:** `WALK-FORWARD · N trades since YYYY-MM-DD · X dropped to warmup (252d)` subtitle when `data.mode === 'walk_forward'`.

Follow this pattern for 3b/c/d.

## Picking up — concrete first steps on the new machine

1. `cd C:\Personal\Data\spx_analysis_dashboard` (or wherever you clone)
2. `git pull` — confirm latest is `951cc57` (Phase 3a).
3. `python run.py` to start the dashboard locally. Or SSH to the VPS and restart the running process there.
4. Hard-refresh the browser and verify the new `Mode: In-sample | Walk-fwd` toggle appears in the page header, and flipping it changes the primary chart.
5. Ask the user which of 3b / 3c / 3d to tackle first. Default to **3b (System Portfolio aggregate)** if they don't specify.

## Recent commits (for context)

```
951cc57 Phase 3a: page-wide walk-forward toggle, /analyze branch added
00236d8 Walk-forward subtitle: separate universe vs primary-filtered counts
ab61c6f Phase 2: walk-forward toggle on Multi-Metric Correlation Explorer
2c89d1e Threshold Drift / All-Ticker Bins: brute-force DOM reset for 5d default
8558372 Threshold Drift: canonical month-end calendar + one-way SELECT binding
00da704 Threshold Drift: midpoint quantile (fixes B20 plateau) + nextTick default
7460d83 Threshold Drift: drift-ratio aggregation + single-ticker view + default fix
cf7ed3f OI Analysis: Threshold Drift pane (walk-forward Phase 1 of 3)
```

## User context worth knowing

- The user has **124 tickers** in the current dataset (a few missing 2019 data — affects warmup math).
- Default outcome dropdown is `ret_5d_fwd_oc` (user asked for this many times; current init() force-selects it via DOM after Alpine reactivity didn't reliably stick).
- The user prefers terse responses; explain the **why** rather than the **what**. They read the diff.
- For exploratory questions, give a recommendation with the main tradeoff and let them redirect.

## Auto-memory note

This handoff and the project context also live in `C:\Users\kroko\.claude\projects\C--Personal-Data-spx-analysis-dashboard\memory\`. On the new machine, that memory is per-user/per-machine — copy the `memory/` directory across if you want the new session to start with the same context, or just point it at this doc.

---

**TL;DR for the new session:** Phase 3a just landed. Walk-forward is now page-wide and drives `/analyze` + the corr explorer. Phases 3b (System Portfolio), 3c (Signal Scanner), 3d (Score Matrix) remain. Ask the user which to do first. Open issue: a few primary charts (rolling, winrate, dist, trade table, calendar) reportedly stopped rendering — user is monitoring, get DevTools console output when they bring it up.
