# Build Brief — "Ticker Analysis" page

A new page in the existing `spx_analysis_dashboard` project. Purpose: a **single-ticker** view (the existing Factor Analysis page is universe-wide/averaged; this one is one ticker at a time). It lets me look at a collection of metrics for one ticker, see average forward P&L by bin and where today's value sits, highlight the extreme-bin dates on a full-history price chart, and inspect the option chain (OI/volume) by strike, by DTE, and over time.

This brief describes intended behavior and layout. A visual mockup (`layoutD.html`) accompanies it — treat the mockup as the **design/interaction reference**, this document as the **spec of record**. Where they differ, ask me.

---

## 0. Scope guardrails

- **This is additive.** Do NOT modify the Factor Analysis page, Score Matrix, Signal Survey, System Portfolio, Secondary Scanner, or any existing endpoint's behavior. New page, new route, new endpoints, new JS module, new CSS scoped to this page.
- Reuse existing infrastructure where it already does the job: the binning/analyze machinery, the `daily_features` table, the DuckDB-over-parquet pattern for raw chain/OI, the cache-table pattern, and the `?v=` JS cache-bust convention.
- **Performance caution.** This box has hit HTTP 502s before from concurrent heavy queries saturating the process. The chain/OI views read raw parquet via DuckDB, which is heavier than the metric panes. Cache aggressively (see §7). Never fire a parquet query on every slider tick without debouncing.
- Track the work in whatever task list you normally use; keep commits scoped (one concern per commit) as we've done before.

---

## 1. Nav & routing

- Add a nav entry labeled **"Ticker Analysis"** in the top nav bar, next to the existing pages.
- Route/page served the same way the other pages are (same templating/static-serving pattern already in the project).
- The page takes a ticker selection (dropdown of the same ticker universe the Factor Analysis page uses). Default to the first ticker or a remembered last-selected — match whatever the existing pages do.
- JS served with the existing `?v=` cache-bust versioning so I don't get stale bundles.

---

## 2. Theme

Use the project's **core theme** exactly as the existing pages do — the charcoal panels, the **bright blue** primary accent and **bright pink** for negatives/puts/drawdown, the stat-strip styling, segmented toggles, and expand icons. Do not introduce a new palette. The mockup approximates these colors; use the real theme variables/classes from the existing stylesheet, not the mockup's hardcoded hexes.

- Positive / calls / long-side / above-level → bright blue.
- Negative / puts / short-side / below-level → bright pink.
- Keep it visually consistent with Factor Analysis so this reads as the same app.

---

## 3. Page structure (top to bottom)

1. **Control bar** — ticker selector, forward-return horizon selector (`ret_1d_fwd_oc`, `ret_3d_fwd_oc`, `ret_5d_fwd_oc`, `ret_7d_fwd_oc`, `ret_10d_fwd_oc`, `ret_20d_fwd_oc`, and the `_cc` variants), a Confluence/Union shade toggle, Clear selection, and Save layout (see §6). Bins are **fixed at 20** — no bin-count selector on this page.
2. **Dynamic stat strip** (see §4) — updates from the currently selected bins.
3. **Full-width price chart** (see §5) — full history always shown.
4. **"Today — what's unusual"** readout row — one cell per metric: today's value, its 20-bin index, and its percentile. Sorted by distance from the median bin so extremes surface first.
5. **Metric panes** grid (see §5) — each pane = two sub-panes.
6. **Option chain section** (see §5.4) — 3D surface, strike×DTE heatmap, multi-mode flow map, OI profile / ΔOI / vol-vs-OI, IV smile + term structure.

---

## 4. Dynamic stat strip (IMPORTANT — changed behavior vs mockup)

The stat strip at the top must be **driven by the current bin selection**, not by the full ticker history.

**Definition of the working set:**
- Each selected bin (across ALL metric panes) contributes a set of `trade_date`s.
- The working set is the **union / dedup** of those dates — a `trade_date` that appears in multiple selected bins is counted **once**.
- If **no bins are selected**, the strip shows stats over the full available history for the selected ticker (sensible default), with a label indicating "all dates."

**Stats to compute over the working set** (using the currently selected forward-return horizon as the outcome):
`Tickers` (always 1 here, or drop it), `N` (count of dates in the working set), `Avg Ret`, `Median`, `Std Dev`, `P5`, `P95`, `Win %`, `# Win`, `Avg Win`, `Avg Loss`, `Trd/Yr`. Match the exact stat set and formatting the Factor Analysis stat strip already uses so they're consistent.

**Recompute trigger:** whenever the selected-bin set changes (any pane), OR the horizon changes, OR the ticker changes. This should be fast — it's an aggregation over a date-filtered slice of `daily_features` for one ticker, so it can be done client-side from data already fetched (preferred, to avoid a round-trip per click) or via a lightweight endpoint if the per-date returns aren't already in the browser.

Color the values with the theme (blue for favorable, pink for P5/Avg Loss etc.) exactly as Factor Analysis does.

---

## 5. Visual components

### 5.1 Full-width price chart
- **Always show the full available history** for the ticker (~7 years). No zoom/pan requirement; the full span is the point. (If you add zoom later it must default to full span.)
- Line chart of close. Mark splits (from `underlying_ohlc.splits`) and, if available, earnings dates.
- **Bin-highlight overlay:** when bins are selected in the metric panes, shade the `trade_date`s belonging to selected bins.
  - **Confluence mode (default):** shade every selected date, opacity scaled by *how many* selected bins that date falls into (more overlap = darker). This makes multi-metric confluence visible as the darkest bands.
  - **Union mode:** flat single-opacity shade for any selected date.
  - Use the bright-blue accent for the shade.
- **Metric overlays:** each metric pane has an "on price" checkbox (§5.2). When checked, draw that metric's value series as a line over the price chart, normalized to its own min/max, in a distinct color, with a small label. Support a few simultaneous overlays with distinct colors.

### 5.2 Metric pane (two sub-panes each)
Each pane has a metric-selector dropdown (any of the ~150 `daily_features` metric columns) and renders **two stacked sub-panes**:

**Sub-pane A — 20-bin average forward P&L bars.**
- **Always 20 bins.** Use the existing 20-bin stats the analyze endpoint already returns (`decile_stats_20` / the per-bin bundle) for `(ticker, metric, horizon)`. Do not re-implement binning if the endpoint already provides 20-bin `{lo, hi, avg_ret, n, win_rate, dates}` per bin.
- Bar height = average forward return for that bin; color blue positive / pink negative.
- Bars are **click-to-select** (toggles the bin into the selection that drives the price chart and stat strip). Show selected state clearly (outline/highlight).
- **Today marker with in-bin position (see §5.3).**
- Tooltip per bin: range `[lo, hi]`, avg P&L, win %, n.

**Sub-pane B — metric value over time.**
- Line of the metric's raw value across the full history (aligned in time to the price chart's x-axis is a plus but not required).
- **When one or more bins are selected in THIS pane:** shade the time series relative to the selected bin level(s). Compute the selected level as the midpoint of the selected bins' combined `[min(lo), max(hi)]` range; draw a faint band/line at that level, shade **above** in faint blue and **below** in faint pink, so I can see which historical periods were above vs below the selected extreme. (If multiple non-contiguous bins are selected, shade relative to the overall selected span; keep it simple.)
- Mark today's latest value.

**"On price" checkbox** in the pane header → toggles the §5.1 overlay for this metric.

**Add/remove panes + saved layouts:** see §6.

### 5.3 Today marker — in-bin lean (NEW behavior)
Instead of only drawing an arrow centered over the bin that today's value falls into, position the marker to reflect **where within the bin** today's value sits:

```
frac = (today_value - bin.lo) / (bin.hi - bin.lo)     # clamp to [0, 1]
marker_x = bin_x_left + frac * bin_width
```

- `frac ≈ 0` → marker at the left edge of the bin; `frac ≈ 1` → right edge; `0.5` → center.
- Render as a small downward triangle at `marker_x` **plus** a thin vertical tick descending into (or just above) the bar, so it reads as "today is here, leaning left/right within this bin."
- Keep the bin itself visually identifiable (e.g. subtle outline on today's bin) in addition to the lean marker.
- Tooltip on the marker: today's value + `frac` as a percent ("62% into bin 14").

### 5.4 Option chain section
All of these read the raw **chain_eod** (volume + IV) and **oi_raw** parquet stores via DuckDB server-side (the established pattern), aggregated to JSON. Respect the `daily_features` "as-of" conventions (OI labeled `trade_date` T is prior-session EOD; chain routed via `feature_date`). Apply the universal split adjustments (strike factor for strike-vs-spot, count factor for OI/volume counts) consistent with `build_features` — do not display raw unadjusted strikes against split-adjusted spot.

Components (all filterable by DTE band and, where noted, moneyness):

- **OI-by-strike horizontal profile** — puts (pink, left) vs calls (blue, right) around a spot line. Date slider to scrub history. DTE + moneyness filters.
- **ΔOI-by-strike** — signed build/unwind per strike over a d1/d5/d20 window (use the existing `d*_total_oi_change` semantics; note it's an overnight position change).
- **Volume-vs-OI per-strike** — scatter (OI x, today volume y) with a reference diagonal; flag strikes where vol/OI is high (fresh activity) in the accent color.
- **3D chain surface** — X = strike, Z = `trade_date`, Y = **signed quantity: calls positive, puts negative** (zero-plane = strike axis). Rotatable (orbit) + zoom. Toggle OI / Vol. Aggregated across DTE (with the DTE filter applied). Overlay the spot path on the zero-plane. Use a WebGL lib (Three.js) — must handle mouse-drag orbit and wheel zoom.
- **Strike × DTE heatmap** — full 2D chain at one snapshot: rows = DTE buckets, columns = strikes, color = OI or Vol (toggle), spot marked as a vertical line. Date slider to pick the snapshot.
- **Flow map (strike × time heatmap)** with a mode toggle: **OI**, **Vol**, **Vol/OI** (turnover/divergence — bright = strikes trading hot relative to standing OI), **ΔOI over N days**, **ΔVol over N days**. Include an N-selector for the change modes and a DTE filter. Overlay spot path. Diverging blue/pink color scale for the signed modes; sequential blue for level modes.
- **IV smile (per snapshot)** and **IV term structure over time (7/30/90d)** — these are shapes, not single metrics, so they get dedicated small panels rather than being binned. Pull from the interpolated IV columns / chain as appropriate.

These chain components are the least-certain part of the design — build them behind the DTE filter and date slider, but expect me to iterate on which are most useful. Prioritize the flow map, the strike×DTE heatmap, and the 3D surface; the profile/ΔOI/vol-vs-OI trio already exist conceptually and are lower priority if time is tight.

---

## 6. Metric panes: add/remove + saved layouts

- I can **add** and **remove** metric panes. Each pane remembers its selected metric and its "on price" checkbox state.
- I can **save named layouts** and recall them. A layout = the ordered set of panes (metric + overlay flags) plus which chain components are expanded, if you implement that.
- Persist layouts in Postgres (a small `ticker_analysis_layouts` table: `id, name, layout_json, created_at`), with endpoints to list/save/delete — mirror the existing endpoint style. A "+add" dropdown and a layout selector in the UI.
- Layouts are ticker-agnostic (a layout is a set of metrics/arrangement; applying it to a different ticker just re-queries for that ticker).

---

## 7. Backend / endpoints

Prefer reusing what exists. Likely additions, all under a new sub-route (e.g. `/api/ticker-analysis/...`) to keep them separate from `/api/factor-analysis/...`:

- **Metric panes:** reuse the existing analyze endpoint that returns 20-bin stats for `(ticker, metric, horizon)` including per-bin `dates`. If it doesn't already return the per-bin date lists for a single ticker efficiently, add a single-ticker variant that does. The per-date metric value series (for sub-pane B and the price overlay) and per-date forward returns (for the dynamic stat strip) should come down in a form that lets the stat strip recompute **client-side** on selection change (avoid a server round-trip per bin click).
- **Chain/OI endpoints:** DuckDB-over-parquet, returning aggregated JSON for each chain component, parameterized by `ticker`, `trade_date` (or snapshot index), `dte_min/dte_max`, `moneyness`, and (for flow/3D) a date range. **Cache results in a dedicated cache table** keyed by the query params (follow the existing cache-table + `/invalidate` pattern; add a `ticker-analysis/*/invalidate` endpoint so cache resets stay scriptable and symmetric with the others). Pre-aggregate strike-level and strike×DTE profiles so the sliders read from cache, not live parquet, after first compute.
- **Layouts:** list / save / delete as in §6.
- Support a `force=1` cache-bust param on the cacheable GETs, consistent with the existing analyze endpoints.

---

## 8. Open decisions to confirm with me before/while building

1. **Bin definition — universe-wide vs per-ticker.** Do the 20 bins use the same universe-wide bin edges as the Factor Analysis page (so "bin 14" means the same thing across pages and today's value is extreme *relative to all tickers*), or per-ticker quantiles (extreme *for this name*)? This affects every P&L pane and the today-marker. My leaning: confirm before building; it may end up a toggle, but pick one default first.
2. **Stat-strip recompute location** — client-side from pre-fetched per-date returns (my preference for responsiveness) vs a lightweight endpoint. Confirm the per-date payload size is reasonable for one ticker over full history; if not, fall back to an endpoint.
3. **Chain component priority** — if time-boxed, build flow map + strike×DTE + 3D surface first; profile/ΔOI/vol-vs-OI second; IV smile/term last.
4. **Split-adjustment reuse** — confirm you're applying the same `make_split_factors` strike/count factors `build_features` uses, rather than re-deriving, so the chain views reconcile with the metric layer.

---

## 9. Acceptance checks

- Selecting/deselecting bins across multiple panes updates: the price-chart highlight (with confluence opacity), the dynamic stat strip (union/dedup dates), and each pane's own time-series shading.
- The stat strip's `N` equals the count of unique `trade_date`s across all selected bins (verify a case where two selected bins share dates → counted once).
- The today marker sits at the correct fractional position inside its bin (spot-check one metric by hand).
- Bins are 20 everywhere; no path produces 10.
- Chain sliders/toggles read from cache after first compute (no repeated multi-second parquet hits; no 502s under normal clicking).
- Nothing on the Factor Analysis / Score Matrix / System Portfolio pages changed.
- Theme matches the rest of the app (bright blue / bright pink, stat strip, toggles).
