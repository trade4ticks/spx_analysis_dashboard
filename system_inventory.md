ok, `# System Inventory

Complete traced inventory of databases, tables, scripts, endpoints, data
flow, and cleanup candidates. Builds on
`dashboard_tables_to_ui.md` (which maps tables → UI panes) by adding
scripts, every endpoint, and dead/vestigial flags.

Update this when adding a table, script, or endpoint, or when retiring
one of the cleanup candidates flagged below.

---

## 0. Architecture at a glance

**Two Postgres databases**, both accessed via `app/db.py`:

| Pool | Env var | Default DB name | Optional? | Tables it holds |
|---|---|---|---|---|
| `get_pool()` (MAIN) | `DATABASE_URL` | `spx_interpolated` | **required** | IV-surface source (`surface_metrics_*`, `spx_surface`, `spx_atm`, `index_ohlc`), AI Explorer (`ai_explorer_*`), Research v1 + v2 (`research_*`, `research2_*`), MAIN-side `oi_*` tables (Score Matrix, Interaction Matrix, Portfolios) |
| `get_oi_pool()` (OI) | `OI_DATABASE_URL` | `open_interest` | optional (None ⇒ OI features degrade) | Source (`daily_features`, `underlying_ohlc`, `option_oi_surface`), bin tables (`is_bins`, `wf_bins`, `tt_bins`, `tt_thresholds`), all caches, `signals`, `tracked_signals`, `oi_signal_calendar`, `metric_classification`, `corner_scan_*` |

The `oi_*` table-name prefix is historical naming — it does **not** mean the table lives in the OI DB. `oi_score_matrix`, `oi_interaction_matrix`, `oi_research_portfolios`, `portfolio_signals` actually live in the MAIN DB. Authoritative routing is in `research/batch_score.py:327` and `app/routers/oi_portfolios.py:9-11`.

**Router-to-prefix map** (from `app/main.py`):

| Router file | URL prefix | DB pool used |
|---|---|---|
| `meta.py`, `skew.py`, `term.py`, `historical.py`, `concavity.py`, `skew_slope.py`, `term_slope.py`, `raw.py`, `heatmap.py`, `today.py` | `/api/{meta,skew,term,historical,convexity,skew_slope,term_slope,raw,heatmap,today}` | parquet + MAIN |
| `ai_explorer.py` | `/api/ai-explorer` | MAIN + OI (SQL agent routes) |
| `research.py` | `/api/research` | MAIN (results) + OI (tickers/columns) |
| `research2.py` | `/api/research2` | MAIN (results) + OI (tickers) |
| `oi_signals.py` | `/api/factor-signals` | OI |
| `oi_analysis.py` | `/api/factor-analysis` | OI |
| `oi_portfolios.py` | `/api/factor-analysis` (shares prefix) | MAIN (portfolios) + OI (aggregate) |
| `backtest_iv.py` | `/api/backtest-iv` | MAIN (upload metadata) + upload data on disk |
| `ticker_analysis.py` | `/api/ticker-analysis` | OI (daily_features/is_bins/underlying_ohlc) + layouts table |
| `ticker_chain.py` | `/api/ticker-analysis` (shares prefix) | OI (spot/splits) + DuckDB over `/data/{oi_raw,chain_eod}` parquet + chain-cache table |

**Total: 43 unique tables across both DBs.**

---

## 1. All Tables

### 1a. MAIN DB (`spx_interpolated`) — 22 tables

| Table | Category | Grain / PK | Populated by | Read by | UI pane / label | Safe to clear? |
|---|---|---|---|---|---|---|
| `surface_metrics_core` | source | (trade_date, quote_time) | External IV pipeline | `meta.py` columns-catalog/value-trail/today-value/dates/columns/latest; AI Explorer SQL agent | All IV-surface charts (Heatmap, Term, Skew, Concavity, both Slopes, Today, Historical, Backtest IV) | N |
| `surface_metrics_catalog` | config | metric name PK | External pipeline | `meta.py` `/columns-catalog`; AI Explorer | Column metadata (group labels, tooltips, formatting) on IV pages | N |
| `spx_surface` | source | (trade_date, quote_time, dte, put_delta) | External pipeline | `heatmap.py`, `term.py`, `skew.py`, `concavity.py`, `skew_slope.py`, `term_slope.py`, `historical.py`, `meta.py` | Raw IV surface — all Heatmap/Term/Skew/Convexity pages | N |
| `spx_atm` | source | (trade_date, quote_time, dte) | External pipeline | `concavity.py`, `term.py`, `skew_slope.py`, `term_slope.py`, `historical.py` | ATM forward IV substitution on Convexity/Slope pages | N |
| `index_ohlc` | source | (trade_date, quote_time) | External pipeline | `today.py` `/scatter` | "SPX vs VIX scatter" on Today page | N |
| `oi_score_matrix` | derived-scan | (ticker, metric, fwd_ret, mode, cutoff_date) | `research/batch_score.py` (CLI) or POST `/api/factor-analysis/run-batch-score` (background) | GET `/score-matrix`, `/score-matrix/meta`, `/score-matrix/summary`; read by `research/interaction_scan.py` | **"Score Matrix"** — first OI Analysis pane (line 402) | Y |
| `oi_interaction_matrix` | derived-scan | (feat_a, feat_b, fwd_ret, ticker) | POST `/run-2f-scan` (background via `research/interaction_scan.py`) | GET `/interaction-matrix`, `/interaction-detail` | **"2-Factor Interaction Scanner"** — nested in Score Matrix pane (line 649) | Y |
| `oi_research_portfolios` | annotation | id SERIAL | POST `/portfolios` | All `oi_portfolios.py` endpoints | Portfolio dropdown + Portfolio Builder pane | N |
| `portfolio_signals` | annotation | (portfolio_id, signal_id) UNIQUE | POST `/portfolios/{pid}/signals` | `oi_portfolios.py` aggregate endpoint | Portfolio ↔ signal membership | N |
| `ai_explorer_log` (a.k.a. `ai_explorer_query_log`) | annotation (write-only) | id SERIAL | `_log_query` after each AI Explorer call | (none — write-only audit) | internal | N (audit log) |
| `ai_explorer_sessions` | annotation | id UUID | POST `/api/ai-explorer/sessions` | AI Explorer GETs/PATCH/DELETE | AI Explorer page — saved sessions sidebar | N |
| `research_runs` | annotation | id UUID | `research/db.py:save_run`; POST `/research/run`, `/research2/run` | research.py + research2.py list/get/delete | Research page + Research2 page run history | N |
| `research_results` | annotation | id UUID, FK → research_runs | `research/db.py:save_result` | research.py / research2.py `/run/{id}/results` | Result tables on Research pages | N |
| `research_series` | annotation | id UUID, FK → research_runs | `research/db.py:save_series` | Research charts rendering | Equity / rolling-corr series on Research pages | N |
| `research_charts` | annotation | id UUID, FK → research_runs | `research/db.py:save_chart` | `/run/{id}/charts`, `/chart/{id}.png` | PNG previews on Research pages | N |
| `research_followups` | annotation | id UUID, FK → research_runs | POST `/run/{id}/followup` | GET `/run/{id}/followups` | Follow-up Q&A on Research pages | N |
| `research_knowledge` | annotation | id SERIAL | POST `/research2/knowledge` | `_load_active_rules` (Research2 + AI summary injection) | "Knowledge Library" on Research2 page; also injected into OI Analysis AI summaries | N |
| `research_pnl_uploads` | annotation | id UUID | POST `/research2/upload-pnl` | GET `/research2/uploads`; `research/orchestrator.py` | "P&L Uploads" picker on Research2 page | N |
| `research_backtest_uploads` | annotation | id UUID | POST `/research2/finalize-backtest` | GET `/research2/backtest-uploads`; `backtest_iv.py` endpoints; orchestrator | "Backtest Uploads" on Research2 page AND Backtest IV Analysis page | N |
| `research_backtest_staging` | derived-cache | id SERIAL | POST `/research2/upload-backtest` (per-chunk) | POST `/research2/finalize-backtest` (consumes) | internal (transient between upload and finalize) | Y |
| `research2_results`, `research2_followups`, `research2_charts`, `research2_runs` | annotation | (Research2's mirror of v1) | `research/db.py` v2 path | research2.py endpoints | Research2 page | N |

### 1b. OI DB (`open_interest`) — 21 tables

| Table | Category | Grain / PK | Populated by | Read by | UI pane / label | Safe to clear? |
|---|---|---|---|---|---|---|
| `daily_features` | source | (ticker, trade_date) | External pipeline (NOT in repo) | Every OI router; all `scripts/*.py`; AI Explorer & Research SQL agents | Every OI Analysis chart + OI Signals + Portfolio Builder + AI Explorer | N |
| `underlying_ohlc` | source | (ticker, trade_date) | External pipeline | AI Explorer agent context; `research/{orchestrator,engine,agent}.py` | OHLC joins in Research/AI Explorer — not directly rendered | N |
| `option_oi_surface` | source | (ticker, trade_date, expiration, strike, option_type) | External pipeline | AI Explorer SQL agent (schema prompt); Research routing | AI Explorer / Research only — not rendered as a dashboard pane | N |
| `is_bins` | derived-bin | (ticker, trade_date) + `bin20_{metric}` cols | **External pipeline (NOT in repo)** | `oi_analysis.py` (heatmap, decile, bundle, /metric-bins, zone-analyze, signals stats); `oi_portfolios.py`; `oi_signals.py`; `corner_scan.py --mode in_sample`; `measure_secondary_corr_bins.py`; `tt_bin_tie_diff.py` | Backs every IS-mode chart (Heatmap, Decile Stats, All-Ticker Metric Bins IS, Corner Scan IS, Portfolio aggregate) | N |
| `wf_bins` | derived-bin | (ticker, trade_date) + `bin20_{metric}` cols | **External pipeline (NOT in repo)** | `oi_analysis.py` (WF bundle, /metric-bins, threshold-drift); `corner_scan.py --mode walk_forward`; `row_compute.py` | Heatmap (WF), Threshold Drift, Corner Scan (WF), All-Ticker Metric Bins (WF) | N |
| `tt_bins` | derived-bin | (ticker, trade_date) + `bin20_{metric}` cols | **External pipeline (NOT in repo)** | `oi_analysis.py` (TT bundle, tt_cutoff, /metric-bins); `tt_bin_tie_diff.py` | TT-mode versions of every binned chart | N |
| `tt_thresholds` | derived-bin | (ticker, metric) thresholds frozen at cutoff | **External pipeline (NOT in repo)** | Read implicitly via tt_bins | TT-mode chart math; not directly rendered | N |
| `metric_classification` | config | metric TEXT PK | `scripts/load_metric_classification.py` (from `daily_features_data_dictionary.md`) | `oi_analysis.py` (family grouping, eligibility); `app/metric_filter.py` | Metric dropdown grouping/labels; eligibility gate | Y (rebuild via script) |
| `sec_scan_cache` | derived-cache | structural_key TEXT PK (FIFO cap 50) | `_write_sec_scan_cache` on POST `/secondary-scan` | GET `/secondary-scan/*` (read-through) | **"Secondary Metrics by Lift"** bar chart + drill-down (line ~2360) | Y |
| `analyze_primary_cache` | derived-cache | cache_key TEXT PK (LRU 2 GB) | `_write_analyze_primary_cache` on GET `/analyze` | GET `/analyze` (read-through) | Heatmap, Distribution by Decile, Decile Stats, Equity by Decile, Yearly bars, Rolling Correlation, Trade Calendar, Secondary nomination | Y |
| `analyze_cache_slim` | derived-cache | cache_key TEXT PK (part of 3-table bundle) | `_write_analyze_bundle_cache` on POST `/analyze-bundle/refresh` (ALL-mode bg) | GET `/analyze-bundle/payload` | Outcome-switched charts + Gap Mode equity (ALL-mode) | Y |
| `analyze_cache_trade_meta` | derived-cache | cache_key TEXT PK | same txn as slim | GET `/analyze-bundle/trade-meta` | Flat trade-data table on bundle outcome switch | Y |
| `analyze_cache_outcome` | derived-cache | (cache_key, outcome) | same txn as slim | GET `/analyze-bundle/outcome` | Per-outcome returns for the 12-outcome bundle | Y |
| `global_bins_cache` | derived-cache | cache_key TEXT PK | GET `/global-metric-bins` (lazy); POST `/global-metric-bins/invalidate` clears | GET `/global-metric-bins` (read-through) | **"All-Ticker Metric Bins"** — fourth OI Analysis pane (line 1306) | Y |
| `ic_batch_cache` | derived-cache | cache_key TEXT PK | `scripts/precompute_ic_all.py` (ALL); GET `/ic-batch` lazy single-ticker; POST `/ic-batch/refresh` | GET `/ic-batch`, `/ic-decomp` | **"Signal Survey"** — fifth OI Analysis pane (line 1422) | Y |
| `corner_scan_2f` | derived-scan | (P, S, dir, outcome, mode) PK | `scripts/corner_scan.py --mode {wf\|is}` | GET `/corner-scan/2f` | **"Corner Scan — 2-Factor"** — second top-level pane (line 902) | Y |
| `corner_scan_1f` | derived-scan | (metric, extreme, outcome, mode) PK | `scripts/corner_scan.py` Phase 5 | GET `/corner-scan/1f` | **"Corner Scan — 1-Factor"** — third top-level pane (line 1135) | Y |
| `corner_scan_notes` | annotation | (P, S, dir, outcome) PK | POST `/corner-scan/notes` | LEFT JOIN in GET `/corner-scan/2f` | Note/reviewed/saved flags in 2F Corner Scan pane | N |
| `signals` | annotation (+ cached cols) | id SERIAL + agg_avg_ret, agg_n, per_cell_stats, stats_updated_at | POST `/signals`, PUT `/signals/{id}` (user); POST `/signals/refresh` (stats only) | GET `/signals`; `oi_signals.py` firing checks; `oi_portfolios.py` membership | Saved cell-set/zone defs; Heatmap Save Signal, OI Signals tracked/firing, Portfolio Builder | N (rows); Y (cached cols via `/signals/refresh`) |
| `tracked_signals` | annotation | signal_id PK | POST `/api/factor-signals/tracked` | `oi_signals.py` `/firing`, `/roster` | OI Signals page — Tracked Signals watchlist | N |
| `oi_signal_calendar` | annotation | id SERIAL, partial-unique (ticker, outcome, entry_date) | POST `/api/factor-signals/calendar` | `oi_signals.py` `/calendar` | "Open Positions Calendar" (Gantt) on OI Signals page | N |
| `ticker_analysis_layouts` | annotation | id SERIAL, name UNIQUE | POST `/api/ticker-analysis/layouts` (upsert by name) | GET/DELETE `/api/ticker-analysis/layouts` | Ticker Analysis page — saved metric-pane layouts (metric+onPrice per pane, shade, horizon); ticker-agnostic. Created lazily by `ticker_analysis.py`. | N (user-entered) |
| `ticker_analysis_chain_cache` | derived-cache | cache_key TEXT PK | read-through on GET `/api/ticker-analysis/chain/*`; force=1 rewrites | same chain GETs | Ticker Analysis option-chain views (profile, ΔOI, vol-oi, strike×DTE, flow, surface, IV smile/term). Created lazily by `ticker_chain.py`; cleared via POST `/chain/invalidate`. | Y |

### 1c. In-memory only — no Postgres table

| Feature | Where | Note |
|---|---|---|
| Threshold Drift cache | `oi_analysis.py` in-memory dict | Sixth OI Analysis pane (line 1729); cleared on server restart |
| `_analyze_bundle_running`, `_analyze_bundle_status` | `oi_analysis.py:4725-4726` | Background-job tracking — Python sets/dicts |
| `_ic_batch_running`, `_ic_batch_status`, `_IC_DECOMP_CACHE` | `oi_analysis.py:4648-4650` | Same pattern |
| `_GLOBAL_BINS_CACHE` (in-memory mirror) | `oi_analysis.py:4260` | Memoization on top of `global_bins_cache` table; `/global-metric-bins/invalidate` clears both |
| `_columns_catalog_cache` | `meta.py:82` | Memoization of `surface_metrics_catalog` |

### 1d. Possibly-dead legacy tables (check pgAdmin)

| Table | Status | Action |
|---|---|---|
| `analyze_cache` | v5 legacy; replaced by `analyze_cache_slim`/`_trade_meta`/`_outcome` (v6) in March 2025. Comments at `oi_analysis.py:4681-4683` say "Drop the legacy table manually after confirming v6 works." | Check if exists; DROP if present |
| `oi_signal_triggers`, `oi_research_systems`, `oi_research_system_library` | Auto-dropped on startup by `oi_signals.py` migration steps | Already cleaned up automatically — verify they're gone |

---

## 2. All Scripts

| File | Purpose | Reads | Writes | Invoked by | Status |
|---|---|---|---|---|---|
| `run.py` | Launches FastAPI app via uvicorn on HOST/PORT | `.env` | — | Manual: `python run.py` on VPS (per deployment memory) | active |
| `run_research.py` | CLI runner for the research-agent workflow | MAIN DB + OI DB + `research_config.yaml` | MAIN DB (`research_runs`, results, charts) | Manual CLI | active |
| `export_report.py` | Loads a completed research run; exports a PDF | MAIN DB (`research_runs`, results, charts) | `reports/<name>.pdf` | Manual CLI | active |
| `scripts/corner_scan.py` | Offline corner-scan: emits 1F + 2F extreme-corner stats | OI DB (`metric_classification`, `wf_bins`/`is_bins`, `daily_features`) | OI DB (`corner_scan_1f`, `corner_scan_2f`, mode-partitioned) | Manual: `python scripts/corner_scan.py --mode {walk_forward\|in_sample} [--force] [--dry-run]`. No cron found. | active |
| `scripts/precompute_ic_all.py` | Pre-computes ALL-mode IC batch one metric at a time (avoids OOM that killed live ALL-mode path) | OI DB (`daily_features` one metric/outcome at a time; `metric_classification`) | OI DB (`ic_batch_cache`, key `ic_batch:ALL:<outcome>:<window>:<mode>:s<stride>`) | Manual: `python scripts/precompute_ic_all.py [--force] [--cutoff-date YYYY-MM-DD]` | active |
| `scripts/load_metric_classification.py` | Parses data dictionary; upserts `metric_classification` | `daily_features_data_dictionary.md` | OI DB (`metric_classification`) | Manual; recovery plan in `dashboard_tables_to_ui.md` | active |
| `scripts/regression_check.py` | Refactor harness: capture JSON snapshots of API output, diff two dirs, train_test_check property tests | Live dashboard HTTP | `regression_snapshots/<tag>/*.json` | Manual during refactor steps; snapshot dirs already present (`step1-baseline`, `step2-after`, …, `step7k`) | active (refactor tool) |
| `scripts/ic_compute_check.py` | DB-free correctness tests for IC primitives (ε derivation, monotone, regime flip, cross-sectional 5-day, sign-stability) | Pure-Python (no DB) | None (prints PASS/FAIL) | Manual: `python scripts/ic_compute_check.py` | active (unit-test suite) |
| `scripts/measure_analyze.py` | Timing harness for `/analyze` bundle compute path; `--verify` asserts parallel vs serial IC are byte-identical | OI DB (`daily_features`) | None (stdout) | Manual on VPS during perf tuning | one-off-diagnostic |
| `scripts/measure_secondary_corr_bins.py` | EXPLAIN ANALYZE: narrow vs wide primary-bin filter on `is_bins` to gate Group-4 migration | OI DB (`is_bins`, `daily_features`, `metric_classification`) | None | Manual on VPS | one-off-diagnostic |
| `scripts/tt_bin_tie_diff.py` | Step-7k one-shot: walks every numeric `daily_features` column under `searchsorted` `side='right'` vs `'left'`; reports any TT-bin shifts at ties | OI DB (`daily_features`, `metric_classification`) | None | Manual; "Run on the VPS once" per docstring. Bug it audits has been fixed. | one-off-diagnostic (cleanup candidate) |
| `research/batch_score.py` | Exhaustive batch-scorer: every ticker × metric × fwd-return; results upserted into `oi_score_matrix`. Has both `__main__` CLI and background-task entry | OI DB (`daily_features`, `wf_bins`, `metric_classification`) | MAIN DB (`oi_score_matrix`) | POST `/api/factor-analysis/run-batch-score` (BackgroundTask) OR `python -m research.batch_score [--walk-forward]` | active |

### Script notes

- **NO bin-builder script in this repo.** Verified by grepping `CREATE TABLE.*(is\|wf\|tt)_bins` and `build_bins|populate_bins|fill_bins`. The bin tables and `tt_thresholds` are populated by an external pipeline (likely the Open_Interest sibling repo). If those tables get truncated, recovery requires running the external pipeline — not anything in this repo.
- **`tt_bin_tie_diff.py`** is the strongest cleanup candidate (the `side='right'` bug it audited has since been fixed in code). Last touched 2026-06-07. Decision: archive vs delete.
- The two `measure_*.py` perf scripts could be archived after the row_compute refactor lands fully, but are still useful if a perf question arises.

---

## 3. All Endpoints

Grouped by router. Status: **active** = wired to JS or polled; **internal-only** = ops use via curl (correct, keep); **VESTIGIAL** = no caller found, deletion candidate.

### 3a. IV-surface routers (`/api/{meta,skew,term,historical,convexity,skew_slope,term_slope,raw,heatmap,today}`)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/api/meta/dates` | parquet | `init()` in heatmap.js, app.js, today.js | Date pickers | active |
| GET | `/api/meta/quote_times` | parquet | init / date-change | Time picker | active |
| GET | `/api/meta/latest` | parquet | (none found) | (none) | **VESTIGIAL** |
| GET | `/api/meta/grid` | parquet | (none found) | (none) | **VESTIGIAL** |
| GET | `/api/meta/columns-catalog` | R: `daily_features` | backtest_iv_analysis.js | Backtest-IV column dropdown semantics | active |
| GET | `/api/meta/value-trail` | R: `daily_features` | backtest_iv_analysis.js | Backtest-IV S1 heatmap × trail | active |
| GET | `/api/meta/today-value` | R: `daily_features` | backtest_iv_analysis.js | Backtest-IV today marker | active |
| GET | `/api/skew/{by_dte,by_date,intraday}` | parquet | app.js dataset switcher | IV Analysis - Skew charts | active |
| GET | `/api/term/{by_delta,by_date,intraday}` | parquet | app.js | IV Analysis - Term charts | active |
| GET | `/api/historical` | parquet | app.js | IV Analysis - Historical | active |
| GET | `/api/convexity` | parquet | app.js | IV Analysis - Convexity | active |
| GET | `/api/skew_slope` | parquet | app.js | IV Analysis - Skew slope | active |
| GET | `/api/term_slope` | parquet | app.js | IV Analysis - Term slope | active |
| GET | `/api/raw/{expirations,skew,term,historical}` | parquet | today.js, app.js | Today + IV Analysis raw mode | active |
| GET | `/api/heatmap/{iv,skew,term,node_stats}` | parquet/aggregate | heatmap.js `loadGrid` | Heatmap page (mode-switched) | active |
| GET | `/api/today/{iv_grid,scatter}` | parquet | today.js | Today page IV grid + scatter | active |

### 3b. AI Explorer (`/api/ai-explorer`)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| POST | `/query` | R: `daily_features` etc. (generated SQL). W: `ai_explorer_log` | ai_explorer.js `submitQuery` | AI Explorer Ask box | active |
| GET | `/sessions` | R: `ai_explorer_sessions` | ai_explorer.js `loadSessions` | Session sidebar | active |
| POST | `/sessions` | W: `ai_explorer_sessions` | ai_explorer.js `saveSession` | "Save session" button | active |
| GET | `/sessions/{id}` | R: `ai_explorer_sessions` | ai_explorer.js `loadSession` | Click session in sidebar | active |
| PATCH | `/sessions/{id}` | W: `ai_explorer_sessions` | ai_explorer.js rename/update | Rename | active |
| DELETE | `/sessions/{id}` | W: `ai_explorer_sessions` | ai_explorer.js `deleteSession` | Delete | active |

### 3c. Research v1 (`/api/research`)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/runs` | R: `research_runs` | research.js `loadRuns` | Run list | active |
| GET | `/run/{id}` | R: `research_runs` + counts | `loadRun` + poller | Run header | active |
| GET | `/run/{id}/results` | R: `research_results` | `loadRun` | Results pane | active |
| GET | `/run/{id}/charts` | R: `research_charts` | `loadRun` | Charts pane | active |
| GET | `/chart/{chart_id}.png` | R: `research_charts.png_data` | `<img src>` via `chartUrl()` | Inline thumbnails | active |
| POST | `/run` | W: `research_runs` + bg writes | `submitRun` | Run button | active |
| DELETE | `/run/{id}` | W: `research_runs` (cascade) | `deleteRun` | Delete-run | active |
| POST | `/run/{id}/retry` | W: research tables | `submitRun` retry path | Retry button | active |
| GET | `/run/{id}/followups` | R: `research_followups` | `loadRun` | Follow-up pane | active |
| POST | `/run/{id}/followup` | W: `research_followups` | `submitFollowup` | Follow-up box | active |
| GET | `/run/{id}/pdf` | reads multiple | `window.location.href` | PDF button | active |
| GET | `/tickers` | R: `daily_features` (OI pool) | `loadMeta` | Ticker picker | active |
| GET | `/columns` | R: information_schema (OI pool) | `loadMeta` | Column picker | active |

### 3d. Research v2 (`/api/research2`)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/runs` | R: `research2_runs` | research2.js `loadRuns` | Run list | active |
| GET | `/run/{id}` | R: `research2_runs` | `loadRun` + poller | Run header | active |
| GET | `/run/{run_id}/charts` | R: `research2_charts` | (none found) | (none) | **VESTIGIAL** |
| DELETE | `/run/{id}` | W: `research2_runs` | `deleteRun` | Delete | active |
| GET | `/run/{id}/pdf` | reads | `downloadPdf` | PDF button | active |
| POST | `/run` | W: `research2_runs` | `submitRun` | Run button | active |
| GET | `/tickers` | R: `daily_features` | `loadMeta` | Ticker dropdown | active |
| GET | `/columns` | R: information_schema | (none found — research.js wires `/columns` only for v1) | (none) | **VESTIGIAL** |
| POST | `/upload-pnl` | W: `research_pnl_uploads` | `uploadPnl` | Upload P&L button | active |
| GET | `/uploads` | R: `research_pnl_uploads` | (none — superseded by `/backtest-uploads`) | (none) | **VESTIGIAL** |
| POST | `/upload-backtest` | W: `research_backtest_staging` | `uploadBacktest` | Upload Backtest button | active |
| POST | `/finalize-backtest` | W: `research_backtest_uploads`, staging | `finalizeBacktest` | (auto after upload) | active |
| DELETE | `/clear-backtest-staging` | W: `research_backtest_staging` | research2.js clear handler | Cancel/cleanup | active |
| GET | `/backtest-uploads` | R: `research_backtest_uploads` | `loadBacktestUploads` | Backtest upload selector | active |
| GET | `/run/{id}/results` | R: `research2_results` | `loadRun` | Results pane | active |
| GET | `/run/{id}/followups` | R: `research2_followups` | `loadRun` | Follow-up pane | active |
| POST | `/run/{id}/followup` | W: `research2_followups` | `submitFollowup` | Follow-up box | active |
| GET | `/knowledge` | R: `research_knowledge` | `loadKnowledge` | Knowledge editor | active |
| POST | `/knowledge` | W: `research_knowledge` | `addKnowledge` | "Add" button | active |
| PATCH | `/knowledge/{id}` | W: `research_knowledge` | `updateKnowledge`/toggle | Edit/toggle | active |
| DELETE | `/knowledge/{id}` | W: `research_knowledge` | `deleteKnowledge` | Delete | active |

### 3e. OI Signals (`/api/factor-signals`)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/tracked` | R: `tracked_signals`, `signals` | (none found — `/roster` and `/firing` cover the read paths) | (none) | **VESTIGIAL** |
| POST | `/tracked` | W: `tracked_signals` | oi_signals.js `addTrackedSignal` | "Add" next to signal dropdown | active |
| POST | `/tracked/from-portfolio` | R: `portfolio_signals`; W: `tracked_signals` | `addTrackedFromPortfolio` | Add-from-portfolio button | active |
| DELETE | `/tracked/{id}` | W: `tracked_signals` | `untrackSignal` | "Untrack" row action | active |
| GET | `/firing` | R: `signals`, `tracked_signals`, `daily_features`, bins | `loadFiring` | "Today's Firings" pane | active |
| GET | `/roster` | R: same | `loadRoster` | Roster table | active |
| GET | `/calendar` | R: `oi_signal_calendar` | `loadCalendar` | Open-positions Gantt | active |
| POST | `/calendar` | W: `oi_signal_calendar` | `addToCalendar` | "Add to calendar" row action | active |
| DELETE | `/calendar/{cid}` | W: `oi_signal_calendar` | `removeFromCalendar` | Gantt delete | active |

### 3f. OI Analysis (`/api/factor-analysis`) — analysis surface

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/tickers` | R: `daily_features` | `init` | Ticker picker | active |
| GET | `/columns` | R: `daily_features`, `metric_classification` | `init` | Metric/Outcome pickers | active |
| GET | `/analyze` | R: `daily_features`, bins; R/W: `analyze_primary_cache` | `loadAnalysis` | Analyze button — entire primary panel | active |
| GET | `/trades` | R: `daily_features`, bins | `_renderTradeTableFlat` (ALL not-cached branch) | Trade-data table | active |
| GET | `/trades/csv` | R: `daily_features` | `exportTradeCSV` (`a.href`) | Export CSV button | active |
| GET | `/heatmap` | R: `daily_features`, bins | `loadHeatmap`, recall path | Heatmap pane | active |
| GET | `/metric-bins` | R: `daily_features`, bins | `loadHmBins1d` | Heatmap 1-D side bars | active |
| GET | `/ai-summary` | R: cached analyze data | `generateAISummary` | "AI Summary" button | active |
| GET | `/score-matrix` | R: `oi_score_matrix` | `loadScoreMatrix` | Score Matrix pane | active |
| GET | `/score-matrix/meta` | R: `oi_score_matrix` | `smInit`, `loadScoreMatrix` | Score Matrix header | active |
| GET | `/score-matrix/summary` | R: aggregate | `loadSmSummary` | Cell tooltip / detail | active |
| POST | `/run-batch-score` | W: `oi_score_matrix` | `runBatchScore` | "Run Scan" button | active |
| GET | `/batch-score-status` | in-memory | `_smStartPoll`, `smInit` | Status spinner | active |
| GET | `/feature-clusters` | static | `loadClusters` | 2F interaction setup | active |
| POST | `/run-2f-scan` | W: `oi_interaction_matrix` | `run2fScan` | "Run 2F Scan" button | active |
| GET | `/2f-scan-status` | in-memory | `_ifStartPoll` | 2F status spinner | active |
| GET | `/interaction-matrix` | R: `oi_interaction_matrix` | `loadInteractionMatrix` | 2F results table | active |
| GET | `/interaction-detail` | R: `oi_interaction_matrix`, `daily_features` | `loadInteractionDetail` | 2F drill-down | active |
| GET | `/tt-cutoff` | R: `tt_bins` | `init` | Page-load metadata | active |
| POST | `/secondary-load` | R: `daily_features`, `sec_scan_cache` | `loadSecondary` | Secondary "Load" button | active |
| POST | `/secondary-scan` | R: `daily_features`; W: `sec_scan_cache` | `runSecondaryScan` | "Load All Metrics" button | active |
| POST | `/secondary-score-status` | in-memory | `_secStartPoll` | Secondary progress | active |
| POST | `/secondary-prepare-rows` | R/W: in-memory `_SEC_CACHE` | `loadSecondary`, corr-bins prep | Chart prep | active |
| POST | `/secondary-scan/invalidate` | W: `sec_scan_cache` | (none) | ops curl | internal-only |
| POST | `/secondary-detail` | R: `daily_features` | `loadSecondaryDetail` | Drill panel | active |
| POST | `/secondary-corr-bins` | R: `daily_features` | `loadCorrMinis` | Corr-bins minis | active |
| POST | `/secondary-correlation` | R: `daily_features` | `loadCorrTable` | Corr table | active |
| POST | `/secondary-zone-analyze` | R: `daily_features`, `signals`, bins | `runZoneAnalyze`, recall path, ALL-mode path | "Zone Analyze" button + recall + ALL mode | active |
| GET | `/global-metric-bins` | R: `daily_features`, bins; R/W: `global_bins_cache` | `loadGlobalBins` | All-Ticker Metric Bins pane | active |
| POST | `/global-metric-bins/invalidate` | W: `global_bins_cache` | (none) | ops curl | internal-only |
| GET | `/global-metric-bins/meta` | R: `global_bins_cache` | `_loadInitMetadata` | Pane header | active |
| GET | `/analyze-bundle` | R: bundle 3-tables | `loadAnalyzeBundle` | Outcome dropdown / Gap toggle gating | active |
| GET | `/analyze-bundle/payload` | R: `analyze_cache_slim` | `_fetchAnalyzeBundlePayload` | Bundle body | active |
| GET | `/analyze-bundle/trade-meta` | R: `analyze_cache_trade_meta` | `_fetchTradeMeta` | Flat trade table source | active |
| GET | `/analyze-bundle/outcome` | R: `analyze_cache_outcome` | `_fetchBundleOutcome` | Outcome switcher | active |
| POST | `/analyze-bundle/refresh` | W: bundle 3-tables | `_refreshAnalyzeBundle` | Hidden refresh trigger | active |
| POST | `/analyze-cache/invalidate` | W: bundle 3-tables | (none) | ops curl | internal-only |
| POST | `/analyze-primary/invalidate` | W: `analyze_primary_cache` | (none) | ops curl | internal-only |
| GET | `/ic-batch` | R: `daily_features`, bins; R/W: `ic_batch_cache` | `loadIcBatch` | Signal Survey pane | active |
| POST | `/ic-batch/refresh` | W: `ic_batch_cache` | `refreshIcBatch` | "Refresh" button | active |
| POST | `/ic-batch/invalidate` | W: `ic_batch_cache` | (none) | ops curl | internal-only |
| GET | `/ic-decomp` | R: `daily_features` | `_loadIcDecomp` | IC decomposition chart | active |
| GET | `/threshold-drift` | in-memory cache | `loadThresholdDrift` | Threshold Drift pane | active |
| POST | `/threshold-drift/invalidate` | in-memory | (none) | ops curl | internal-only |
| GET | `/threshold-drift/meta` | in-memory | `_loadInitMetadata` | Pane header | active |
| GET | `/corner-scan/meta` | R: corner_scan_2f/1f | `_loadInitMetadata`, `loadCs2f`, `loadCs1f` | Pane headers | active |
| GET | `/corner-scan/2f` | R: `corner_scan_2f`, `corner_scan_notes` | `loadCs2f` | Corner Scan 2F pane | active |
| POST | `/corner-scan/notes` | W: `corner_scan_notes` | `cs2fSaveNote`, `cs2fSaveReviewed`, `cs2fSaveSaved` | Row actions | active |
| GET | `/corner-scan/1f` | R: `corner_scan_1f` | `loadCs1f` | Corner Scan 1F pane | active |
| GET | `/signals` | R: `signals` | `loadSavedSignals`, recall flow | "Recall signal"/save dropdown | active |
| POST | `/signals` | W: `signals` | `saveSignal`, recall override-create | "Save Signal" button | active |
| DELETE | `/signals/{id}` | W: `signals` | (none — bulk endpoint used instead) | (none) | **VESTIGIAL** |
| PUT | `/signals/{id}` | W: `signals` | recall update path | Recall + rename flow | active |
| POST | `/signals/refresh` | W: `signals` (stats cols only) | `refreshSignalStats` | "Refresh stats" button | active |
| POST | `/signals/delete-batch` | W: `signals` | `deleteSignalBatch` | Bulk-delete flow | active |

### 3g. OI Portfolios (`/api/factor-analysis` — shares prefix)

| Method | Path | Tables R/W | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/portfolios` | R: `oi_research_portfolios` | `loadPortfolios`, oi_signals.js `loadPortfoliosForAdd` | Portfolio dropdown (both pages) | active |
| POST | `/portfolios` | W: `oi_research_portfolios` | `createPortfolio` | "+ Portfolio" button | active |
| PUT | `/portfolios/{pid}` | W: `oi_research_portfolios` | `renamePortfolio` | Rename | active |
| DELETE | `/portfolios/{pid}` | W: portfolios + cascade `portfolio_signals` | `deletePortfolio` | Delete | active |
| GET | `/portfolios/{pid}` | R: portfolios + signals | `loadPortfolioDetail` | Detail pane | active |
| POST | `/portfolios/{pid}/signals` | W: `portfolio_signals` | `addSignalToPortfolio` | Add-signal button | active |
| DELETE | `/portfolios/{pid}/signals/{ps_id}` | W: `portfolio_signals` | `removeSignalFromPortfolio` | Remove row | active |
| PUT | `/portfolios/{pid}/signals/{ps_id}` | W: `portfolio_signals` | `updatePortfolioSignal` | Edit weight | active |
| POST | `/portfolios/{pid}/aggregate` | R: `portfolio_signals`, `signals`, `daily_features` | `runPortfolioAggregate` | "Aggregate" button | active |

### 3h. Backtest IV Analysis (`/api/backtest-iv`)

All POST endpoints take the upload_id; analytics over the upload data.

| Method | Path | JS caller | UI | Status |
|---|---|---|---|---|
| GET | `/uploads` | `loadUploads` | Upload selector | active |
| GET | `/{upload_id}/columns` | `loadColumns` | Column picker | active |
| GET | `/{upload_id}/column-stats` | `_loadColumnStats` | Stats overlay | active |
| POST | `/{upload_id}/summary` | `_api('summary')` | S0 pane | active |
| POST | `/{upload_id}/heatmap` | `_api('heatmap')` | S1 heatmap | active |
| POST | `/{upload_id}/delta-r2` | `_api('delta-r2')` | S2 pane | active |
| POST | `/{upload_id}/decile` | `_api('decile')` | S3 multi-metric decile | active |
| POST | `/{upload_id}/conditional-slice` | `_api('conditional-slice')` | S4 pane | active |
| POST | `/{upload_id}/distribution` | `_api('distribution')` | S5 pane | active |
| POST | `/{upload_id}/time-stability` | `_api('time-stability')` | S6 pane | active |
| POST | `/{upload_id}/feature-correlation` | `_api('feature-correlation')` | S7 pane | active |
| GET | `/{upload_id}/top-bottom` | `_api('top-bottom')` | S8 pane | active |
| GET | `/{upload_id}/correlation-overview` | direct fetch | Correlation overview chart | active |

### 3i. Ticker Analysis (`/api/ticker-analysis`) — `ticker_analysis.py` + `ticker_chain.py`

Single-ticker page. Metric layer reads OI DB (Postgres); chain views read
`/data/{oi_raw,chain_eod}/{TICKER}/{YEAR}.parquet` via DuckDB with split
adjustment vendored in `app/split_factors.py`, cached in
`ticker_analysis_chain_cache`. All chain GETs take `force=1` (cache-bust)
and, where applicable, `dte_bands`/`moneyness`/`side=all|call|put`.

| Method | Path | Tables / source | JS caller | UI | Status |
|---|---|---|---|---|---|
| GET | `/tickers` | R: `daily_features` | init | Ticker selector | active |
| GET | `/price` | R: `underlying_ohlc` | loadPrice | Full-history price chart | active |
| GET | `/metric` | R: `daily_features`, `is_bins` | loadPaneData | Metric pane (20-bin IS bars + value series + today) | active |
| GET | `/today-scan` | R: `daily_features`, `is_bins` | loadTodayScan | "Today — what's unusual" row | active |
| GET | `/layouts` | R: `ticker_analysis_layouts` | loadLayouts | Layout selector | active |
| POST | `/layouts` | W: `ticker_analysis_layouts` (upsert by name) | saveLayout | Save layout | active |
| DELETE | `/layouts/{id}` | W: `ticker_analysis_layouts` | deleteLayout | Delete layout | active |
| GET | `/chain/dates` | R: `daily_features` (total_oi not null) | loadChainDates | Chain date slider | active |
| GET | `/chain/oi-profile` | parquet `oi_raw`; R/W chain-cache | loadChainProfile | OI-by-strike profile | active |
| GET | `/chain/doi-profile` | parquet `oi_raw` | loadChainDoi | ΔOI-by-strike | active |
| GET | `/chain/vol-oi` | parquet `oi_raw`+`chain_eod` | loadChainVolOi | Vol/OI ratio-by-strike | active |
| GET | `/chain/strike-dte` | parquet `oi_raw`\|`chain_eod` | loadChainHeatmap | Strike×DTE heatmap (OI/Vol) | active |
| GET | `/chain/flow` | parquet `oi_raw`+`chain_eod` (multi-year) | loadChainFlow | Flow map (strike×time; OI/Vol/V-OI/ΔOI/ΔVol) | active |
| GET | `/chain/surface` | parquet `oi_raw`\|`chain_eod` | loadChainSurface | 3D surface (Three.js; Z=time\|dte) | active |
| GET | `/chain/iv-smile` | parquet `chain_eod` | loadChainSmile | IV smile (per snapshot) | active |
| GET | `/chain/iv-term` | parquet `chain_eod` (multi-year) | loadChainIvTerm | IV term structure (7/30/90d) | active |
| POST | `/chain/invalidate` | W: `ticker_analysis_chain_cache` | (none) | ops curl (optional `?ticker=`) | internal-only |

New files: `app/split_factors.py` (split factors vendored from Open_Interest/lib/split_factors.py), `app/routers/ticker_chain.py`. Page route `GET /ticker-analysis` → `templates/ticker_analysis.html` (loads Three.js r128 + OrbitControls from CDN for the 3D surface).

---

## 4. Data Flow

Text outline, not a diagram. Indentation = data dependency.

### 4a. External pipeline (NOT in this repo)

The Open_Interest sibling repo (or similar) populates these sources daily/weekly:

- **OI DB sources** — `daily_features`, `underlying_ohlc`, `option_oi_surface`
- **OI DB bin tables** — `is_bins`, `wf_bins`, `tt_bins`, `tt_thresholds` (built from `daily_features`)
- **MAIN DB IV-surface sources** — `surface_metrics_core`, `surface_metrics_catalog`, `spx_surface`, `spx_atm`, `index_ohlc`

If any of these are missing, the dashboard sections that read them degrade or fail. The IV-surface pages stay isolated from the OI side — they share neither sources nor caches.

### 4b. Inside-this-repo flow (OI Analysis side)

```
[External: daily_features] + [External: is_bins / wf_bins / tt_bins / tt_thresholds]
  │
  ├──→ scripts/load_metric_classification.py
  │       └──→ metric_classification (config; gates metric eligibility)
  │
  ├──→ scripts/corner_scan.py --mode walk_forward / in_sample
  │       └──→ corner_scan_2f, corner_scan_1f
  │              └──→ GET /corner-scan/{2f,1f} → Corner Scan panes
  │              └──→ LEFT JOIN with corner_scan_notes (preserved across reruns)
  │
  ├──→ scripts/precompute_ic_all.py --force
  │       └──→ ic_batch_cache (ALL-mode key)
  │              └──→ GET /ic-batch → Signal Survey pane
  │       Single-ticker /ic-batch fills the same cache lazily.
  │
  ├──→ research/batch_score.py (CLI or POST /run-batch-score)
  │       └──→ oi_score_matrix (MAIN DB)
  │              └──→ GET /score-matrix → Score Matrix pane
  │
  ├──→ research/interaction_scan.py (POST /run-2f-scan)
  │       └──→ oi_interaction_matrix (MAIN DB)
  │              └──→ GET /interaction-matrix → 2F Interaction Scanner
  │
  ├──→ Lazy caches (filled on first read, served thereafter):
  │       analyze_primary_cache  ← GET /analyze            → primary analyze panes
  │       global_bins_cache      ← GET /global-metric-bins → All-Ticker Metric Bins
  │       sec_scan_cache         ← POST /secondary-scan    → Secondary Metrics by Lift
  │       analyze_cache_slim +   ← POST /analyze-bundle/refresh (ALL) or
  │       _trade_meta +              GET /analyze-bundle (single-ticker)
  │       _outcome                                            → outcome-switched / Gap mode
  │
  └──→ In-memory only: threshold-drift cache, bundle compute status
          └──→ GET /threshold-drift → Threshold Drift pane
```

### 4c. Annotation attachment points

User-entered data attaches to specific tables via the UI:

- **Saved cell-sets** ("zones") → `signals` rows from POST `/signals` in the Heatmap pane's "Save Signal" flow. Re-association: `oi_signal_calendar`, `tracked_signals`, `portfolio_signals` all reference `signals.id`.
- **Corner-scan notes** → `corner_scan_notes` from POST `/corner-scan/notes`. Re-associate to scan reruns via LEFT JOIN on `(primary_metric, secondary_metric, corner_direction, outcome)`.
- **Portfolios** → `oi_research_portfolios` + `portfolio_signals` (cascade-deletes).
- **Calendar entries** → `oi_signal_calendar` (manual Gantt entries).
- **Watchlist** → `tracked_signals`.
- **Research runs** → `research_runs` + child tables (v1) or `research2_*` (v2).
- **Knowledge rules** → `research_knowledge` (Research2 page; also injected into OI Analysis AI summaries).
- **Uploads** → `research_pnl_uploads`, `research_backtest_uploads` (file content stored in DB).
- **AI Explorer sessions** → `ai_explorer_sessions`.

### 4d. OI Signals page flow

```
signals (annotation) + tracked_signals (annotation)
  ├──→ GET /api/factor-signals/firing → "Today's Firings" pane
  ├──→ GET /api/factor-signals/roster → roster table
  └──→ GET /api/factor-signals/calendar (+ oi_signal_calendar) → Gantt
```

### 4e. Research v1 / v2 flow

```
POST /research/run or /research2/run
  └──→ background research.orchestrator
        ├── reads: daily_features (OI), research_knowledge (MAIN), uploads (MAIN)
        └── writes: research_runs, research_results, research_series, research_charts, research_followups (MAIN)
              └──→ Research page run-detail panes
```

### 4f. IV-surface side (completely independent of OI side)

```
[External: surface_metrics_*, spx_surface, spx_atm, index_ohlc]
  └──→ heatmap.py, term.py, skew.py, concavity.py, *_slope.py, historical.py, today.py
        └──→ Heatmap page + IV Analysis page + Today page (no cache tables; parquet/SQL direct)
```

### 4g. Ticker Analysis side (single-ticker page)

```
Metric layer (Postgres, OI DB):
  [daily_features] + [is_bins] + [underlying_ohlc]
    └──→ ticker_analysis.py (/tickers, /price, /metric, /today-scan)
          └──→ price chart + metric panes + "what's unusual" + stat strip
    └──→ ticker_analysis_layouts (annotation)  ← /layouts CRUD

Chain layer (DuckDB over parquet — NOT Postgres):
  [/data/oi_raw/{TICKER}/{YEAR}.parquet] + [/data/chain_eod/{TICKER}/{YEAR}.parquet]
    └──→ ticker_chain.py (/chain/*), split-adjusted via app/split_factors.py
          (vendored from Open_Interest/lib/split_factors.py; splits from
           underlying_ohlc.splits; spot from daily_features.spot_pc = prior close)
          └──→ read-through cache: ticker_analysis_chain_cache  (POST /chain/invalidate clears)
          └──→ OI profile / ΔOI / vol-oi / strike×DTE / flow / 3D surface / IV smile / IV term
```

The chain parquet stores are the SAME ones the external `build_features`
pipeline reads (`data/{oi_raw,chain_eod}` there); this page reads them
directly at runtime on the VPS via `OI_RAW_DIR` / `CHAIN_EOD_DIR` env
(default `/data/...`).

---

## 5. Dead / Vestigial Summary — Cleanup Candidates

### 5a. Vestigial endpoints (no JS caller, no internal use)

| Endpoint | Reason | Risk |
|---|---|---|
| `GET /api/meta/latest` | No caller anywhere | Low |
| `GET /api/meta/grid` | No caller anywhere | Low |
| `GET /api/research2/run/{id}/charts` | research2 chart access flows through `/results` instead | Low — verify no `<img>` tag uses it before deleting |
| `GET /api/research2/columns` | Copy-paste from v1 router; v2 only wires `/tickers` | Low |
| `GET /api/research2/uploads` | Superseded by `/backtest-uploads` after the UI consolidated on backtests | Low |
| `GET /api/factor-signals/tracked` | List of tracked signals is served by `/roster` and `/firing`; raw `/tracked` listing has no caller | Low |
| `DELETE /api/factor-analysis/signals/{id}` | All deletes flow through `/signals/delete-batch` (single + bulk) | Low |

### 5b. Cleanup-candidate scripts

| Script | Reason |
|---|---|
| `scripts/tt_bin_tie_diff.py` | One-shot audit of the `side='right'` tie bug; bug has since been fixed in code. Useful only for re-running the audit. Archive or delete. |
| `scripts/measure_analyze.py`, `scripts/measure_secondary_corr_bins.py` | One-shot perf-decision tooling. Keep if you anticipate more perf-tuning sessions; otherwise archive. |

### 5c. Possibly-dead legacy tables to check in pgAdmin

| Table | Why it might exist | Action |
|---|---|---|
| `analyze_cache` | v5 legacy of the bundle cache; replaced by `analyze_cache_slim`/`_trade_meta`/`_outcome` (v6). Code comment at `oi_analysis.py:4681-4683` says "Drop manually after confirming v6 works." | Check pgAdmin; `DROP TABLE IF EXISTS analyze_cache` if present |
| `oi_signal_triggers`, `oi_research_systems`, `oi_research_system_library` | Auto-dropped on every startup by `oi_signals.py` migration steps | Should already be gone; verify |

### 5d. Things that look suspect but are NOT vestigial

| Looks dead, but isn't | Why it stays |
|---|---|
| The `/invalidate` POST endpoints with no JS caller | Used by ops via curl; documented in `dashboard_tables_to_ui.md` |
| `ai_explorer_log` writes with no reader | Write-only audit log by design |
| `scripts/ic_compute_check.py` | DB-free unit-test suite for IC primitives |
| `scripts/regression_check.py` | Active refactor harness; snapshots used across the row_compute migration |

### 5e. Documentation drift fixed in this pass

- `dashboard_tables_to_ui.md` previously listed `POST /api/factor-analysis/analyze-bundle/invalidate` as a "convenience alias." That route does NOT exist in any router. The invalidate cascade is exposed only via `/analyze-cache/invalidate`. The line has been removed.

---

## Coverage gaps you'll need to fill yourself

These can't be answered from inside this repo:

1. **External bin builder** — no script in this repo creates `is_bins`, `wf_bins`, `tt_bins`, or `tt_thresholds`. Document the upstream pipeline's location and the CLI used to rebuild.
2. **External IV-surface pipeline** — same story for `surface_metrics_*`, `spx_surface`, `spx_atm`, `index_ohlc`. Document where these are built.
3. **VPS cron jobs (if any)** — no crontab references in repo. If `corner_scan.py` or `precompute_ic_all.py` are scheduled on the VPS, that scheduling is external. Document in the deployment notes if so.
4. **Possibly-dead tables in DB** — verify whether `analyze_cache`, `oi_signal_triggers`, etc. still exist in your DB. The code drops them on startup but a fresh pgAdmin pass would confirm.
