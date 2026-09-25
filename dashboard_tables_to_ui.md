# Dashboard Tables → UI Mapping

Reference for decoding internal Postgres table names into the dashboard
panes/buttons the user actually sees. Use when deciding what to clear,
what to preserve, and what regenerates automatically.

Sibling doc to `daily_features_data_dictionary.md`. Update this when
adding a new cache table or moving a UI pane.

**Current to 2026-09-24.** The three sections below the original ones cover
the tables added since 2026-08-16 — the backtest pages' saved work, the
Equity IV and Equities Scalp tables, and the live service's files, which are
not in Postgres at all.

---

## Cache / derived tables (the ones you most often want to clear)

| Table | Populated by | UI pane / label | Where it sits | Clearing affects | Safe to clear? |
|---|---|---|---|---|---|
| `oi_score_matrix` | GET `/api/factor-analysis/score-matrix` (lazy); POST `/api/factor-analysis/run-batch-score` (explicit rebuild) | **"Score Matrix"** | First major pane on OI Analysis page (line 402) | Score Matrix table goes blank. Auto-rebuilds on next "Run Scan" click (~3–10 min). | Y |
| `oi_interaction_matrix` | POST `/api/factor-analysis/run-2f-scan`; read by GET `/interaction-matrix`, `/interaction-detail` | **"2-Factor Interaction Scanner"** | Nested inside the Score Matrix pane (line 649) | Interaction Scanner results table empties. Auto-rebuilds on next "Run 2F Scan" click. | Y |
| `corner_scan_2f` | `scripts/corner_scan.py --mode {wf\|is\|tt}` | **"Corner Scan — 2-Factor"** | Second top-level pane (line 902) | Table empties. Re-run the script to regen — once per mode. | Y |
| `corner_scan_1f` | Same script (Phase 5) | **"Corner Scan — 1-Factor"** | Third top-level pane (line 1135). Separate collapsible pane, not a sub-tab of 2F | Table empties. Re-run the script. | Y |
| `global_bins_cache` | GET `/api/factor-analysis/global-metric-bins` | **"All-Ticker Metric Bins"** | Fourth top-level pane (line 1306). 6-column grid of 20-bin avg-return profiles | Pane goes blank; click "Refresh" — ~2–5 s per outcome to rebuild. | Y |
| `ic_batch_cache` | `scripts/precompute_ic_all.py` (ALL-mode); GET `/api/factor-analysis/ic-batch` (single-ticker lazy); POST `/api/factor-analysis/ic-batch/refresh` (explicit) | **"Signal Survey"** (IC + breadth + stability leaderboard) | Fifth top-level pane (line 1422) | Pane shows "not_ready" placeholder. ALL-mode needs explicit Refresh / precompute re-run (~30 s – 2 min). See gotcha #1 below. | Y |
| (in-memory only — no DB table) | GET `/api/factor-analysis/threshold-drift` | **"Threshold Drift"** | Sixth top-level pane (line 1729). WF-only; IS/TT show "pending in-sample build" placeholder | In-memory dict clears (also clears on server restart). Click Refresh to rebuild (~2–5 s). | Y |
| `sec_scan_cache` | POST `/api/factor-analysis/secondary-scan` (background) | **"Secondary"** scanner — "Secondary Metrics by Lift" bar chart and drill-down | Inside the primary analyze flow under the Heatmap (line ~2360) | Lift bar empties. Click "Load All Metrics" / "Reload All" to rebuild (~1–3 min). | Y |
| `analyze_primary_cache` | GET `/api/factor-analysis/analyze` (lazy fill on every Analyze click; LRU-capped at 2 GB) | **All of the primary analyze visuals**: Heatmap, Distribution by Decile, Decile Stats table, Equity by Decile, Yearly bars, Rolling Correlation, Trade Calendar, plus the Secondary nomination chart | Body of OI Analysis page (after Ticker / Metric / Outcome / Analyze) | Next Analyze re-fires; computation takes longer (~3–30 s depending on universe) before charts appear. No functional loss. | Y |
| `analyze_cache_slim` + `analyze_cache_trade_meta` + `analyze_cache_outcome` (3-table set, cleared together) | GET `/api/factor-analysis/analyze-bundle` and `/analyze-bundle/payload` / `/trade-meta` / `/outcome`; POST `/analyze-bundle/refresh` for ALL-mode background fill | Outcome-switched charts when user picks an outcome ≠ default `ret_5d_fwd_oc`; **Gap Mode** equity charts; flat trade-data table | Driven by the outcome dropdown + Gap Mode toggle in the decile section | Next outcome switch / Gap toggle triggers background bundle compute (~30 s – 2 min for ALL-mode; single-ticker recomputes inline at /analyze and never uses these tables) | Y |
| `ticker_analysis_chain_cache` (OI DB) | Read-through on GET `/api/ticker-analysis/chain/*` (oi-profile, doi-profile, vol-oi, strike-dte, flow, surface, iv-smile, iv-term); `force=1` rewrites | **Ticker Analysis** page → Option Chain section (all views) | The DuckDB-over-parquet chain views | Chain views recompute from `/data/{oi_raw,chain_eod}` parquet on next open (heavier than the metric layer — seconds). No functional loss. Clear via POST `/api/ticker-analysis/chain/invalidate`. | Y |

---

## Structural / source / annotation tables (for full context — DO NOT clear)

| Table | What it holds | Safe to clear? | Why |
|---|---|---|---|
| `daily_features` | Source: per-(ticker, trade_date) OHLC + computed metrics | **N** | Source data; rebuild requires re-ingesting parquet/upstream |
| `underlying_ohlc` | Source: SPX spot reference | **N** | Source |
| `is_bins` / `wf_bins` / `tt_bins` | Stored `bin20_{metric}` per (ticker, trade_date) for IS / WF / TT modes | **N** | Every decile-based chart breaks (Heatmap, Decile Stats, Equity-by-Decile, All-Ticker Bins, Corner Scan, Bundle pre-bin fields, Threshold Drift). Rebuilt by an external pipeline outside this repo. |
| `tt_thresholds` | Per-ticker, per-metric thresholds frozen at cutoff date | **N** | TT-mode charts break |
| `metric_classification` | family num / family name / tier / eligibility per metric | Mostly Y (UX degrades, computations still work) | Metric dropdowns lose grouping. Reload with `scripts/load_metric_classification.py --dict-path daily_features_data_dictionary.md --verify` |
| `corner_scan_notes` | User notes, reviewed, saved flags per (P, S, direction, outcome) | **N** | User-entered. Re-associates to a fresh `corner_scan_2f` via LEFT JOIN on the 4-column key. The corner-scan script never touches this table. |
| `signals` | User-saved cell-set definitions (name, primary_metric, secondary_metric, outcome, n_bins, cell_set) PLUS cached stats columns (agg_avg_ret, agg_n, per_cell_stats, stats_updated_at) | **N** for rows | POST `/api/factor-analysis/signals/refresh` only refreshes the four cached-stats columns; never touches the user-entered columns. |
| `tracked_signals` | User watchlist | **N** | User-entered |
| `oi_research_portfolios` | Portfolio metadata | **N** | User-entered |
| `portfolio_signals` | Portfolio ↔ signal membership | **N** | User-entered (cascade-deletes only on portfolio delete) |
| `oi_signal_calendar` | Manual signal-firing calendar entries | **N** | User-entered |
| `research_pnl_uploads` | Uploaded P&L CSVs | **N** | User uploads |
| `research_backtest_uploads` | Uploaded backtest sets | **N** | User uploads |
| `research_knowledge` | AI Explorer rules | **N** | User-entered |
| `ticker_analysis_layouts` (OI DB) | Saved Ticker Analysis metric-pane layouts (name, ordered panes: metric + "on price" flag, shade mode, horizon) — ticker-agnostic | **N** | User-entered (Ticker Analysis "Save layout"). Created lazily; upsert by name. |

---

## Invalidate endpoints (clear cache without recompute)

All return `{"ok": true}` or `{"ok": true, "deleted": N, "scope": "..."}`. No auth on Tailscale-local.

| Endpoint | Clears | Notes |
|---|---|---|
| POST `/api/factor-analysis/secondary-scan/invalidate` | `sec_scan_cache` | — |
| POST `/api/factor-analysis/global-metric-bins/invalidate` | `global_bins_cache` + in-memory dict | — |
| POST `/api/factor-analysis/analyze-primary/invalidate` | `analyze_primary_cache` | Optional `?ticker=&metric=&outcome=` scopes (all / ticker / ticker+metric / ticker+metric+outcome) |
| POST `/api/factor-analysis/analyze-cache/invalidate` | `analyze_cache_slim` + `trade_meta` + `outcome` (cascade) | Optional `?ticker=&metric=` scopes |
| POST `/api/factor-analysis/ic-batch/invalidate` | `ic_batch_cache` | — |
| POST `/api/factor-analysis/threshold-drift/invalidate` | in-memory only | — |
| POST `/api/ticker-analysis/chain/invalidate` | `ticker_analysis_chain_cache` | Optional `?ticker=` scope (matches keys containing `:TICKER:`) |

For corner-scan and 2F-interaction caches, there's no invalidate endpoint — those are rebuilt by re-running the script (`scripts/corner_scan.py`) or hitting the explicit Run/Refresh button in the UI (which writes new rows on top of the existing ones).

---

## Notable distinctions

### 2-Factor Interaction Scanner vs. Corner Scan 2F
Two different scans, two different tables, two different UI panes:

- **2-Factor Interaction Scanner** (`oi_interaction_matrix`): ranks all metric-pair *combinations* across all tickers/forward-returns. Reports interaction lift (avg benefit when both metrics co-fire). Lives **inside** the Score Matrix pane. Triggered by "Run 2F Scan" button.
- **Corner Scan — 2-Factor** (`corner_scan_2f`): finds specific decile-extreme combinations (e.g. high metric A + high metric B). Decile-level extremes, not pair-interaction scoring. **Separate top-level pane**. Built by `scripts/corner_scan.py`.

If you want to clear "the 2F thing," check which pane is showing the data first — they're independent and use different SQL.

### Bundle 3-table set
`analyze_cache_slim` + `analyze_cache_trade_meta` + `analyze_cache_outcome` are FK-related normalized children. Always clear via `/analyze-cache/invalidate` (or `/analyze-bundle/invalidate`) — the endpoint cascades the DELETE in one transaction. **Do not TRUNCATE individual children directly**; you'll desynchronize the set.

### Threshold Drift is WF-only
The IS and TT mode tabs aren't broken — they intentionally show "pending in-sample build" placeholder text. Not vestigial.

---

## Gotchas

1. **IC ALL-mode double-stride bug** — the live inline `/ic-batch` ALL-mode path computes IC with effective window = `window × stride` (252 × 3 = 756 days instead of 252). The precompute script is correct. **Always re-run `scripts/precompute_ic_all.py --force` after clearing `ic_batch_cache`**; don't rely on the lazy lookup path to rebuild the ALL entry.

2. **Schema-version salts auto-invalidate** — most caches use keys like `ap:v{N}:`, `ab:v{N}:`, `sv:v{N}:`. Bumping a `_*_SCHEMA_VERSION` constant on the next deploy makes old rows unreachable; the table self-prunes on startup. No manual invalidate needed in that case.

3. **`analyze_primary_cache` LRU at 2 GB** — large `/analyze` payloads (full SPX 20-year history, all-ticker heatmap) evict older rows quickly. Expected behavior; no maintenance action needed.

4. **`signals` table — `/signals/refresh` is the right tool** — it updates `agg_avg_ret`, `agg_n`, `per_cell_stats`, `stats_updated_at` and leaves `name`, `primary_metric`, `secondary_metric`, `outcome`, `n_bins`, `cell_set`, `created_at` untouched. Don't `DELETE FROM signals` — you'll lose user-saved cell-sets.

5. **`corner_scan_notes` survives `corner_scan.py` reruns** — the script's `INSERT ... ON CONFLICT` targets `corner_scan_2f` only; notes re-associate via LEFT JOIN on `(primary_metric, secondary_metric, corner_direction, outcome)` in the `/corner-scan/2f` endpoint.

---

## Rebuild order after a daily_features change

For reference; full discussion in housekeeping notes / chat history.

1. `is_bins`, `wf_bins`, `tt_bins`, `tt_thresholds` (external pipeline — not in repo)
2. `scripts/load_metric_classification.py` (only if data dictionary changed)
3. `scripts/corner_scan.py --mode walk_forward --force` then `--mode in_sample --force`
4. `scripts/precompute_ic_all.py --force`
5. Cache invalidates (any order): `/secondary-scan/invalidate`, `/global-metric-bins/invalidate`, `/analyze-primary/invalidate`, `/analyze-cache/invalidate`, `/threshold-drift/invalidate`
6. `POST /api/factor-analysis/signals/refresh` with all signal IDs

---

## Saved work — added since 2026-08-16 (these do NOT regenerate)

Everything in the first section is a cache: clear it and it rebuilds. These
hold work that was typed or uploaded, and nothing rebuilds them.

| Table | DB | Populated by | UI pane / label | Clearing affects | Safe to clear? |
|---|---|---|---|---|---|
| `oo_backtest_strategies` | MAIN | POST `/api/oo-backtest/strategies` — the Save box on **OO/Mesosim Backtest** | Saved-strategy dropdown there, and the strategy picker on **Backtest Portfolio** | Every saved trade log is gone. The row holds the **original uploaded file** (`file_gz`) and it exists nowhere else — re-upload the CSV/JSON | **N** |
| ↳ its `parsed_*` columns only | MAIN | `app/oo_backtest/parsed_cache.py`, on first load of each file | (nothing visible) | Speed only: the next load re-parses, ~3 s per log — a five-strategy portfolio pays it five times, ~17 s cold against ~0.2 s warm. Keyed on `file_sha256` + a hash of `data_loader.py`, so it self-invalidates | **Y** (`UPDATE … SET parsed_gz = NULL`) |
| `backtest_portfolio_profiles` | MAIN | POST `/api/backtest-portfolio/profiles` — Save on the profiles card | **Backtest Portfolio** — the saved-profile dropdown | Saved combinations (which strategies, qty, capital, per-strategy filters, date-range mode, rolling window) are gone. The strategies they point at are not | **N** |
| `equity_structure_presets` | OI | POST `/api/equity-iv/structures` | **Equity IV** — saved structure presets | Saved presets are gone | **N** |
| `fills`, `fills_daily` | SCALP | POST `/api/equities-scalp/upload-fills` (a Schwab statement) | **Equities Scalp** — traded markers, realized P/L, the traded columns | Uploaded fills are gone; re-upload the statements | **N** |

## Source tables for the newer pages (external pipelines — DO NOT clear)

None of these is built in this repo, and nothing here can rebuild them.

| Table | DB | Built by | UI pane / label |
|---|---|---|---|
| `equity_metrics`, `equity_metrics_z`, `equity_metrics_catalog` | OI | External equity-IV pipeline | **Equity IV** — every metric pane, the scanner, the cross-section, the Base/Z toggle |
| `equity_atm`, `equity_surface` | OI | External equity-IV pipeline | Equity IV ticker header and rails; all six surface panels (curve band, tent, sticky strike, surface grid, time scatter, spot-vol) |
| `earnings_calendar`, `earnings_coverage` | OI | External equity-IV pipeline | Earnings markers, and the coverage statement made before anything is drawn from them |
| `universe`, `daily_metrics` | SCALP | Open_Interest `scalp/` pipeline | **Equities Scalp** — the scan grid, the filter pane's ranges, every daily chart |
| `intraday_metrics` (~14 days), `intraday_monthly` (kept) | SCALP | same | Scalp **ticker detail**, `scope=sessions` and `scope=months` |
| `provenance` | SCALP | same | The freshness / coverage line on the scalp page |

The SCALP database is reached through an **optional** pool: when it is absent
the page says "not connected" and the rest of the dashboard starts normally.

## Not tables at all — the live service (port 8001)

`spx-live` writes no Postgres. Its state is two things on disk and the tape
in memory.

| Path | Written by | UI | Clearing affects | Safe to clear? |
|---|---|---|---|---|
| `data/wall_watchlist.json` | POST `/wall/watchlist` (temp file + rename) | **Equities Wall** — the symbols and each pane's scale override | The wall comes back empty and the list is retyped by hand | **N** (user-entered) |
| `data/scan_history/scan_cells_<date>.npz` | `live/scan_history.py`, once a minute, one file per session date, **not pruned** | **Equities Scan** — the grid's accumulated minutes | Today's grid is blank until it re-accumulates; past dates stop being loadable | Y (loses history) |
| (memory only) the tape rings | `live/hub.py`, `scan.py`, `wall.py` | All three live pages | Nothing persists — trades and quotes are discarded as they age out of the tier's window. A restart empties every pane **and silently disarms trading** | — |
