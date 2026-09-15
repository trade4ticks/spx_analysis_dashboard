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
bins draw nothing. Avg/total bar **opacity scales with √(count / largest bin)**, floored at
0.25; hue stays profit/loss. Both are cosmetic and meant to be easy to revert. A value outside a fixed category list (e.g. a Saturday) gets its own bar, never
dropped. Three section states, worded differently: **skipped** (the log has no values in
the column), **no data** (the current filter leaves none), **ready**. `check_oo_backtest`
holds JS parity with `calculate_bin_stats` / `calculate_correlation` on the full set and a
filtered subset.

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
  matches `pd.cut` at, just below, and just above every edge of every range metric. Any
  new binning must be covered by that parity gate.
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
