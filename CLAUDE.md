# CLAUDE.md

Durable project knowledge for Claude Code sessions. Other docs at the repo root:
`system_inventory.md` (tables, scripts, endpoints, data flow), `dashboard_tables_to_ui.md`
(table → pane lookup), `DEPLOYMENT_NOTES.md`, `MIGRATION_PRINCIPLES.md`, `NEXT_SESSION.md`.

---

## OO/Mesosim Backtest page (`/oo-backtest`)

Replaces the old Dash/Render backtest app. Code: `app/oo_backtest/` (parsers, `market.py`,
`registry.py`, `payload.py`, `stats.py`, `store.py`), `app/routers/oo_backtest.py`,
`templates/oo_backtest.html`, `static/js/oo_backtest.js`.

Phases: P1 scaffold/parsers/registry (`3c34b28`), P2 market data (`4e509b9`),
P2b saved strategies (`10ec652`), P3 filters/stats/charts (`8e89003`), P4 the ten metric
sections. Remaining: P5 (docs).

**Metric sections (P4).** One card per registry entry with `section: true`: avg and total
P/L by bin, plus P/L vs metric with an OLS line (categorical: no scatter; `winRate` adds a
separate win-rate chart — never a second y-axis). Per-bin values follow
`calculate_bin_stats`, but **range metrics keep empty bins** (deliberately unlike its
`observed=True`) so the axis stays to scale — dropping them put ">40" beside "18–20"; empty
bins draw nothing. **Every bar chart's opacity** (avg, total, win rate, Day of Week) is
`OB_ALPHA_FLOOR + (1 − floor) × (count / largest bin) ^ OB_ALPHA_GAMMA` — 0.12 and 1.0,
named constants for tuning by eye; relative to the chart's largest bin, no absolute
thresholds. Hue stays profit/loss. Both are cosmetic and meant to be easy to revert.

**Binning `fixed` | `auto`** (registry field). Premium is `auto`: the page builds its edges
from the WHOLE loaded log (a filter never moves them) — p1..p99 snapped out to whichever of
$10/$25/$50/$100/$250 gives a bin count nearest 24 (tie → smaller step), outliers in `<lo` /
`≥hi` end buckets (≥ because bins are left-closed). The section header names the step
("$25 bins (auto)"): two logs with different steps are not bar-for-bar comparable. Auto
metrics are excluded from the fixed pd.cut parity; `check_auto_bins` tests them against a
numpy reference instead. Everything else stays `fixed`.

A value outside a fixed category list (e.g. a Saturday) gets its own bar, never dropped. Three section states, worded differently: **skipped** (the log has no values in
the column), **no data** (the current filter leaves none), **ready**. `check_oo_backtest`
holds JS parity with `calculate_bin_stats` / `calculate_correlation` on the full set and a
filtered subset.

**Summary stats, capital, Deployment.** 15 stats, three rows of five: # Trades · Win % ·
Total P/L · Avg P/L · Avg P/L % / Avg Win · Avg Loss · Max Win · Max Loss · Profit Factor /
Avg Annual P/L · Avg Annual Return % · Max DD · Calmar · Avg Days. The added five
(`obExtraStats`): years = (last exit − first entry) / 365.25 over the filtered trades;
Avg Annual P/L = total / years; Calmar = that / |max DD $|; Profit Factor = gross wins /
|gross losses|; Avg Annual Return % = avg annual P/L / (peak concurrency × capital); Avg
P/L % = avg P/L / capital. **Capital per position** is a display input (default $10,000):
it recomputes only those figures and the Deployment chart, no re-filter, no re-parse. It
is stored as `capital_per_position` on `oo_backtest_strategies` (NULL = default; column
added by `ADD COLUMN IF NOT EXISTS`); on a loaded saved strategy a committed change is
written with `PUT /strategies/{id}/capital`, which leaves `updated_at` (the list order)
alone. **Deployment** (a registry `pane`: its own card, on the Day of Week row beside that
section's card, not inside it) counts open positions per
**SPX session** from the rollup (`market.session_days`, sent as `market.spx_sessions`),
entry and exit day both inclusive. Every session in the span is a point, so a stretch with
nothing open is a run of zeros. One stepped line; the right axis is the left × capital,
pinned to the same range, not a second trace. Still-open MesoSim positions are excluded by
the parser and so contribute nothing (gate-checked against the v3.1 fixture). Trades
opening or closing off an SPX session (e.g. 2026-04-08) are counted from the next session
and reported under the chart.

## Live trading: the broker interface (2026-09-17)

A second broker (DAS Trader, CMD API) is coming, so Schwab moved behind an
interface. **`live/broker.py` is the façade** everything talks to — the pane,
`live/main.py`'s `/broker/*` endpoints, the checks. It owns the POLICY: the four
switches, the guards, and that a flatten needs trading allowed. **Adapters live in
`live/brokers/`**: `base.py` (the `Broker` ABC + `BrokerError`/`BrokerIndeterminate`
+ the shared shape-matching `match_placement`), `schwab.py` (all Schwab protocol,
moved verbatim), `__init__.py` (selection by `LIVE_BROKER`, default `schwab`,
unknown value raises rather than falling back).

**The safety property:** arming and the guards are checked ONCE in the façade,
above the adapter, so a new adapter cannot trade while disarmed or past the limits
by forgetting to ask. `check_broker.py` drives a FakeBroker and asserts the adapter
was NOT CALLED AT ALL when a switch or guard refuses, and scans every
`live/brokers/*.py` for policy tokens (`armed=`, `check_guards(`,
`trading_allowed(`, `_runtime_enabled`) so the DAS adapter is held to it too.
Exceptions kept deliberately: `cancel` is never gated on arming; `flatten` needs
trading allowed but not the pane's arm flag.

Adapter internals are now `schwab.*` — the checks and probes monkeypatch
`schwab._acall` / `schwab._account_hash`, not `broker.*`. Rate limits belong to the
adapter (a broker-API fact); the rule they serve — getting flat must never be
refused for quota — stays in the façade's `priority` flags.

### Surface metrics exploration (P6, in progress)

Phases (approved 2026-09-15): **P6a** server groundwork — done; **P6b** ranking chart —
done; **P6c** added metric rows — done; **P6d** per-row filters — done. **P6 is complete**; no
separate doc (user, 2026-09-15) — explanations live in tooltips (BH, Spearman vs Pearson,
common coverage only; `surfTip`, coverage dates and chance counts read from the data). **P6c** add-a-metric rows (generic "nice"-step auto bins, units
formatting); **P6d** per-row filters with row/page scope (row default; page scope shows the
trades a coverage gap would drop); P6e docs.

**`public.surface_metrics_core`** (verified by the data owner): 110,258 rows, 545 MB, 458
DOUBLE PRECISION columns, PK (trade_date, quote_time) + index on trade_date. Clean: no rows
on non-trading days, 78 bars per session 09:35–16:00, no NaN, no zeros — none of the
index_ohlc validity machinery applies. Start-labeled, aligned with index_ohlc (spot at T =
spx_open at T). NULL before a metric's coverage, **never gaps after**, so a missing value
is either "before coverage" or "no bar" (a 09:30 entry). Coverage moves with backfills —
`surface.get_catalog` reads each metric's first non-null date (index walk, ~4 s on the VPS,
cached; a full scan is no faster) and rebuilds when the table's first date, last date **or
row count** changes — dates alone missed a mid-range insert during a live backfill. An
UPDATE into existing rows is not detected. **Never hardcode coverage dates**.
**`surface_metrics_catalog`** is a real table (PK column_name); the repo CSV was diffed
identical on 2026-09-15. Ranked set = catalog minus families `meta`, `spot`, `forward`
(452 of 462); `log_ret` kept. Column names reach SQL only from that set.

**Entry bar: the entry's own bar, no lookahead (confirmed 2026-09-15).** Every metric is
point-in-time at quote_time with backward-looking windows, so `BAR_RULE =
"at_or_before_entry"`, `LOOKAHEAD_CONFIRMED = True`; the `previous_bar` alternative was
removed and a request naming it gets a 400. The bar is on the entry date only — never the
prior session.

Stats (`surface_stats.py`): pairwise n, distinct entry bars (effective sample, reported not
corrected), Pearson and Spearman r + p, Benjamini-Hochberg for both over every metric
computed (a hidden family is still a test). **Vectorised by null pattern**: columns sharing
a null mask (one group per coverage start) form a dense submatrix; P/L is ranked once PER
GROUP (ranking it once globally corrupts late-starting metrics' Spearman — gate-planted);
p from `special.betaincc` (Pearson) and `special.stdtr` (Spearman) as scipy does. VPS: the
per-metric scipy loop took 2.5 s; locally the rewrite does 2,097 × 452 in ~75 ms. Matches
per-metric scipy to r 2e-14 / p 1.6e-12 relative (gate: 1e-12 / 1e-9), not bit-identical.
The join (26 ms, fully cached, PK backward scan) needs no index or column reduction. Endpoints `GET /surface/catalog`,
`POST /surface/rank`, `POST /surface/values` (values fetched for ALL loaded trades, so a row
follows page filters without another call). `scripts/measure_surface_rank.py` is the
read-only VPS timing script for the 458-column join.

**Ranking chart (P6b)**, a card below the sections. Ranked on the server only when *Rank
metrics* / *Recompute* is pressed, for the current filtered trades; a later filter or
coverage change marks it **stale** (signature of the filtered rows + common start), no
auto-request. The **headline states prominently how many trades have a metric bar**, split
into "entered before coverage" and "no bar at the entry time" (on the OO log ~1,348 of
2,097; before ranking it states how many predate the metrics in view). Bars: fixed 12 px,
horizontal scroll, y axis in its own fixed chart beside the scroller; sorted by |value| of
the chosen method (Spearman default / Pearson); form filter; legend toggles single families
(hidden families re-pack; BH still counts them); opacity = `obBarAlpha(n, max n in view)`.
**BH is a faint dashed line after the last bar surviving q 0.05** (sorting method) — no bar
outlines (the user found white borders abrasive). BH is not a pure |r| threshold (adjusted p
depends on n), so survivors need not be a contiguous run: `obBhBoundary` puts the line after
the last survivor. Non-survivors left of it get a **prominent warning box**, not a footer
aside: metrics sharing a coverage start share n, so later-starting forms (z-scores) sit
SYSTEMATICALLY at the low end of n and fail at an |r| that passes for levels — the box names
their forms and n ranges. (An earlier note calling this "rare" was wrong.) **Common coverage only** sends
only trades from the latest `min_date` among metrics in view, and states its cost first:
trades in [earliest start, common start) that some metric in view would lose. Family hues
(`surface.FAMILY_GROUPS`, 8 groups for ~14 families; unassigned → grey "Other") and form
labels come with the catalog response — the page JS names no family (the dropped-scope
guard enforces it). **Reactivity trap, hit twice:** anything derived from data kept outside
the Alpine proxy must read a reactive value or it never re-renders — `OB_DATA.idx` →
`idxTick` (the headline was blank), `OB_DATA.rank` → `surf.result` in `surfView()` (the
footer and BH warning were blank on screen while direct calls, and so the node gates,
returned the right text). Gates call methods directly and cannot see this; only a rendered
check can. The fixed y-axis column needs `min-width:0; overflow:hidden` or its canvas's
intrinsic width widens it (~290 px) and shifts every bar.

**Added metric rows (P6c).** A ranking bar click or the family-grouped dropdown adds a
section row below the ranking card. The row fetches that ONE metric for EVERY trade in the
log (`POST /surface/values`), stores it as client column `surface__<name>` (never in the
payload whitelist), scaled to display units by `surface.UNIT_FORMATS` (vol_decimal ×100 →
vol pts, log_return ×100 → %; unit list sent with the catalog). From then on it is a
registry-shaped entry in `sectionMetrics()` — binned, filtered and drawn by the built-in
section code with no request. Bins: `ROW_AUTO_BINS`, Premium's rule with `steps: "nice"`
(1/2/2.5/5 × 10^k around span/24, edges cleaned to 12 significant digits, label decimals ≥
step's). Header: column, catalog description, n with a value, step, "data from"; OLS slope
is stated per bin step. A duplicate add scrolls to the row; a failed fetch leaves an error
row; a new log refetches every row (a log token discards late responses); remove drops the
column, bins and charts. The section card is one Jinja macro (`metric_section_card`) shared
by built-in and added rows. Rows are not saved with a strategy.

**Per-row filters (P6d).** Each added row with values gets a dual slider (display units,
step = bin step / 10) and a scope toggle. **This row** (default) narrows only that row's
charts — page count, stats, other sections and the active-filter count are untouched.
**Whole page** adds a range spec to `activeSpecs()` like a sidebar filter, so once narrowed
every trade WITHOUT a value is dropped (the old app's Filter Scope trap). While page scope is
selected the row states the cost against the page's other filters: trades with no value,
split "entered before its data starts" / "no bar at the entry time" — before narrowing
("moving the slider will drop N") and while narrowing ("dropping N"). Reset-all resets row
filters (keeps scope); removing a page-scoped row re-filters the page.

The **Market-data checks** pane is collapsed by default (header toggles it).

### What `main.index_ohlc` actually looks like

5-minute bars, SPX/VIX/VIX3M/VIX9D full OHLC, 2017-01-01 → present.

- **Bars are labeled by start time.** 09:30 is the first bar; the 15:55 bar spans
  15:55–16:00, so its close is the session close. 16:00 rows are partial/NaN and never read.
- **Every non-trading day has 78 rows of zeros** (weekends and holidays alike). The table
  is a complete calendar scaffold with real data written only into sessions.
- **Holidays also carry 19–25 "artifact" VIX bars** — stale/partial rows on closed days.
  Not data.
- **Postgres NaN:** NaN sorts above every number. `x <= 0` is false for NaN, `x > 0` is
  true, `x = 'NaN'` is true. A validity test needs `> 0 AND <> 'NaN'` (not null, not NaN,
  positive). Make values valid *before* any aggregate — `max()` over a column with one NaN
  returns NaN. This bit three separate times.
- **Coverage starts:** SPX 2017-01-03, VIX 2017-01-03, VIX3M 2017-10-24, VIX9D 2018-06-08.
- **2026-04-08** is a real trading day (full VIX session) with zero SPX bars — missing
  data, not a non-session.

**Session rule** (`MIN_SESSION_BARS = 34` in `market.py`): a series has a session on a day
when it has ≥ 34 valid bars in 09:30–15:55. Artifact holidays peak at 25; the shortest
real session (early close) is 41. Margins are shown in the Market data detail panel.

Sessions are **per series**. SPX's previous close is from the previous day SPX had a
session; VIX's from the previous day VIX had one. **No trading-day calendar anywhere** —
it would disagree with the table on days like 2026-04-08. The rule is derived from data.

**Known upstream data problems (not page bugs):**
- 2026 16:00 bars stopped: ~249 valid/year through 2025, 65 of 174 in 2026. Ingestion
  changed mid-year. The page uses 15:55, so unaffected.
- VIX hole May–July 2026: 23 session days with no VIX. Causes the 17 null VIX levels and
  clustered null VIX gaps.

### Decisions, and why

- **Filtering happens in the browser.** The full trade set ships once; filter changes
  recompute locally, no server round trip (the old Dash app round-tripped every slider
  nudge). Dates serialize as ISO strings deliberately.
- **Bin edges come from the server; binning happens in JS.** A gate asserts JS binning
  matches `pd.cut` at, just below, and just above every edge of every fixed range metric.
  Any new binning must be covered by that parity gate. The one exception is `binning:
  "auto"` (Premium): the server sends the rule (steps, target, percentiles), the page
  builds the edges from the log, and `check_auto_bins` holds that against numpy.
- **One metric registry** (`registry.py`) drives the filter sidebar, the sections, and the
  payload whitelist. Adding a metric should be one entry (the old app used five parallel
  dicts).
- **SharpTwo and skew are dropped** — columns removed, not just hidden, so no all-NaN
  series reach correlation/binning code. `minDate` on the registry (filled at request time
  from coverage) exists so skew can return without rebuilding the old app's cross-filter
  scope toggles for truncated-coverage metrics.
- **No cross-check against Option Omega's `Gap` column** (removed in `0a222e5`). OO's
  column has a different, undocumented definition (once matching a post-16:00 straggler,
  once not). Hand-verified 2018-05-07: computed 5.64 matches the OHLC exactly. The parser
  does not copy OO's `Gap` into `gap`.
- **No market-data cron.** `index_ohlc` is maintained by something else on the VPS; a
  fetch path would be a second writer. "Update Market Data" is a read-only freshness
  indicator: stale when the latest valid SPX bar is more than `STALE_AFTER_DAYS = 5`
  calendar days old.
- **No materialized view for the daily rollup.** It is a CTE run once per process and
  cached, rebuilt when `index_ohlc`'s latest row moves. A view wouldn't have prevented the
  session-rule bug and would add a shared-DB object needing a refresh owner.
- **Saved strategies store the original uploaded file, not processed trades.** Loading
  re-parses and re-joins, so saved logs never freeze save-day market data (the
  zero-filled-day fix would otherwise have left old saves with wrong Monday gaps).

### Open items

1. Pick a ratio basis (entry-time vs daily-close for `vix3m_vix_ratio` / `vix_vix9d_ratio`),
   then delete the losing columns and the temporary sidebar switch.
2. Confirm the status line reads "Showing 2,097 of 2,097" on the OO CSV (2,064 was the
   pre-fix count; 2,064 would mean untouched filters exclude null-valued trades).
3. Zero-P/L trades count toward Avg Loss but not as a win (matches the old app). Confirm
   whether the logs contain any.

---

## Gates (`scripts/gates.py`)

Split by host (`872530c`). **Nothing reports PASS that it did not run**; under `--deploy`
a gate that could not run is a FAIL.

- **Dev machine:** everything, including `check_oo_backtest` (needs node and the
  Options-Backtest-Dashboard checkout) and `check_oo_market_sql` (needs `initdb`). Run
  before pushing, with `./.venv/Scripts/python.exe`.
- **VPS:** after `git pull`,
  `sudo -u gates .venv/bin/python scripts/gates.py --vps --deploy` — runs only
  `check_routes_smoke` and `check_template_render`. The VPS has no node and no Postgres
  server binaries (the DB is elsewhere), so the SQL gate can never run there.

One-time VPS setup (also in the `gates.py` docstring):

```bash
sudo useradd --system --create-home --shell /bin/bash gates
sudo -u gates git config --global --add safe.directory /spx_analysis_dashboard
sudo chgrp gates /spx_analysis_dashboard/.env && sudo chmod 640 /spx_analysis_dashboard/.env
```

`check_vendored` fails on pre-existing drift in `scalp_config.py` and
`scalp_metric_docs.py` (confirmed at `d7fc7dc`, before the OO work). Unrelated.
