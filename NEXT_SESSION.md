# SPX Dashboard — Session Handoff

Read top-to-bottom on the new machine. Self-contained briefing for picking up where the previous session left off.

## TL;DR — current state (2026-05-22)

Two long-running initiatives now mostly done, one in flight:

1. **row_compute refactor (Steps 1–7)** — *DONE*. Bin computation is centralized in `app/routers/row_compute.py`. The page-wide `Mode: In-sample | Walk-fwd | Train-test` toggle drives every binning surface through one dispatch. All five Step 7 cleanup items resolved (7e–7k). Score Matrix train-test had a real `searchsorted side="right"` bug fixed in 7k; 5 features have contaminated pre-7k data that should be re-scanned (`pct_up_days_20d`, `max_oi_strike_put`, `max_oi_strike_call`, `relative_strength_vs_spy_20d`, `pct_from_52w_high`). Details in the IC and Step 7 sections below.

2. **IC (Information Coefficient) tooling (Steps IC.1–IC.4)** — *IC.4 just landed, visually verified*. New `app/routers/ic_compute.py` sibling module + `/ic-batch` endpoint produces per-metric rolling Spearman IC + sign-stability for all ~123 metrics in any mode. The `/analyze` rolling-IC pane is now populated in both single-ticker and ALL (cross-sectional) mode with mode-aware reference IC, ε noise floor, and per-window sign classification. **Next up: IC.5 — frontend leaderboard + scatter (universe-wide view).**

3. **Step cadence**: one step per session, hard-stop verification (regression diff + sometimes manual eyeball) between each. See the IC plan section and the user-feedback memory note at the bottom.

## The refactor in one paragraph

Pre-refactor, every endpoint had a 2-way `if walk_forward / else` fork plus inline binning math. Adding train/test would have required ~30 new if-branches across 5 endpoints. Now there's one `BinningSpec` (in-sample | walk-forward | train-test), one `make_spec(walk_forward, cutoff_date)` constructor, one `ASSIGNERS` registry, and a handful of free dispatch functions (`filter_by_assignments`, `assign_secondary_bin_stats`, `assign_secondary_buckets`, `secondary_membership`) + one class (`PortfolioVectorBuilder`) that endpoints call. Adding a fourth method (rolling IC, cluster id, anomaly score, etc.) becomes a registry entry plus implementations of each dispatch function — no endpoint surgery.

## The page-wide mode toggle (user-facing)

Top of the OI Analysis page, in the `.sel-bar` row alongside Ticker/Metric/Outcome/From/To: a 3-way `Mode: In-sample | Walk-fwd | Train-test` segmented toggle (`.go-btn`-sized). When `Train-test` is selected, a `Cutoff` date input appears next to the toggle (default `2024-01-01`).

Every binning surface on the page reads from `pageMode` and `cutoffDate`: `/analyze`, All-Ticker Metric Bins, the 2D heatmap (ALL-mode), the heatmap's 1D side bin charts, Multi-Metric Correlation Explorer (mini-bins + result), Secondary Signal Scanner, Secondary Detail, System Portfolio Aggregate. Flipping the toggle triggers `setPageMode(m)` which cascades to every visible fetch site.

Score Matrix is the one binning surface NOT routed through this layer yet — it falls back to in-sample when pageMode is train-test. Flagged in the Step 7 cleanup list.

## Architecture: `app/routers/row_compute.py`

The single source of truth for binning. Contents:

### Contract
- `RowAssignment` dataclass — `{ticker, trade_date, metric_name, metric_value, n_bins, bin, outcome_col, forward_return, dropped_reason}`. `trade_date` is `Any` (preserves asyncpg's `datetime.date`; downstream legacy code does `dd - last_date` on `pair[2]` which needs date objects).

### Spec dataclasses
- `InSampleSpec`, `WalkForwardSpec(warmup=252)`, `TrainTestSpec(cutoff: date, warmup_in_train=252)`. Each has a `kind: Literal[...]` discriminator. `BinningSpec` = union.

### Spec constructor
- `make_spec(walk_forward: bool, cutoff_date: Optional[str]) -> BinningSpec`. Single construction point used by every endpoint. `cutoff_date` wins over `walk_forward`.

### Assigner Protocol + classes
- `RowAssigner` Protocol with `fit`, `assign`, `assign_batch` methods.
- `InSampleAssigner` — wraps `_bucket_pairs_per_ticker` / `_bucket_pairs`. `assign_batch` wraps `_compute_all_bins_fast`.
- `WalkForwardAssigner(warmup=252)` — wraps `_walk_forward_bucket_per_ticker` / `_walk_forward_bucket_pairs`. `assign_batch` wraps `_compute_all_bins_walk_forward`.
- `TrainTestAssigner(cutoff, warmup_in_train=252)` — `fit` builds per-ticker sorted training history from rows with `trade_date < cutoff`. `assign` calls `_bin_for_value(value, frozen_history, n_bins)` for every row. Training-window rows get `bin = K` AND `dropped_reason = "pre_cutoff"` so aggregators skip them while preserving the bin for a future side-by-side view. `assign_batch` delegates to `_compute_all_bins_train_test_fast` (numpy-vectorized via `np.searchsorted`) — without this the batch path on 200K-row × 80-feature data exceeds Cloudflare's 100-second upstream timeout.

### Registry
- `ASSIGNERS: dict[str, type] = {"in_sample": ..., "walk_forward": ..., "train_test": ...}`. Endpoint flow: `spec = make_spec(...); assigner = ASSIGNERS[spec.kind](spec)`.

### Free dispatch functions (used by secondary endpoints + heatmap)
- `filter_by_assignments(rows, spec, primary_metric, selected_primary_bins, is_all, filtered_dates)` → `(filtered_chrono, dropped, universe)`. Primary filter for the four secondary endpoints. Train-test branch excludes `pre_cutoff` rows from the kept set (test-only universe).
- `assign_secondary_bin_stats(spec, rows_chrono, metric, n_bins, outcome_col, is_all)` → `{name, bins, bin_ns}` or `None`. Per-feature stats inside an already-primary-filtered subset.
- `assign_secondary_buckets(spec, rows_chrono, metric, n_bins, outcome_col, is_all)` → `list[list[tuple]]`. Per-bin row-tuple lists for `/secondary-detail`'s downstream equity/yearly/ticker construction.
- `secondary_membership(spec, rows_chrono, metric, selected_bins, n_bins, is_all)` → `np.ndarray` 0/1 vector.

All three train_test branches filter `dropped_reason is None` so training rows don't enter aggregations — the test-window-only semantic ("Option A").

### Portfolio builder
- `PortfolioVectorBuilder(spec, rows, is_all)` — encapsulates the per-system loop for `/portfolios/{pid}/aggregate`. `primary_vector(metric, selected_bins, n_bins)` and `secondary_vector(metric, selected_bins, n_bins, primary_indices)` return 0/1 vectors. Handles the in-sample-vs-walk-forward asymmetry (in-sample bins secondaries on the primary-filtered subset; walk-forward/train-test bins on the full rows then ANDs with the primary mask) and caches per-(metric, n_bins) bin maps across calls. Train-test branch treats `pre_cutoff` rows as bin=None in the cache so they don't fire as system trades.

### Runtime invariants
- `_validate_assignments(assignments, n_bins)` — runs on every `assign()` consumer in dev. Asserts `bin` is `None` or `int in [1, n_bins]`. Kept on permanently.
- `dropped_count_for_mode(spec, assignments) -> int` — mode-aware: 0 for in_sample, count of `"warmup"` for walk_forward, count of `"insufficient_train_history"` for train_test. Used by every response envelope.

## Per-step status

| Step | Migrated | Commit |
|------|----------|--------|
| **1** | `row_compute.py` + `scripts/regression_check.py` (additive dead code) | `9ffc284` |
| **2** | `/analyze` — both ALL and single-ticker branches. Bit-equivalent after two follow-ups (`8f5897b`, `19bfe2f`, `e866848`) that matched legacy bucket-iteration and within-date pair ordering. | `4c70892` + follow-ups |
| **3** | `/global-metric-bins` — Assigners gain `assign_batch` for vectorized multi-feature batch. | `ee92d72` |
| **4** | `/secondary-corr-bins`, `/secondary-correlation`, `/secondary-detail`, `/secondary-scan` — all four routed through `filter_by_assignments` + the three secondary dispatch helpers. `/secondary-detail`'s inline bucket-of-tuples logic moved into `assign_secondary_buckets`. | `7731f61` |
| **5** | `/portfolios/{pid}/aggregate` — per-system loop encapsulated in `PortfolioVectorBuilder` (eliminates 3 `if walk_forward` branches inside the loop). | `080cd99` |
| **5.5** | `/heatmap` ALL-mode (single-ticker stays on `np.percentile` edges by design). | `ce6fc4a` |
| **5.5 continuation** | `/metric-bins` (heatmap side bin charts) + frontend wiring so the heatmap respects pageMode end-to-end. Found via manual UI review: original Step 5.5 migrated `/heatmap`'s 2D grid but missed `/metric-bins`, plus three frontend gaps. | `025ab50` |
| **6** | `TrainTestAssigner` activated end-to-end: every dispatch function gets its train_test branch; every endpoint accepts `cutoff_date`; frontend gets a 3-way toggle + date input. | `e782401` + 3 hotfixes |
| **6 hotfix 1** | TrainTestAssigner preserves original `trade_date` type (not stringified) so `_equity_for_decile` doesn't crash. | `fd93d0c` |
| **6 hotfix 2** | TrainTestAssigner.assign_batch delegates to numpy-vectorized `_compute_all_bins_train_test_fast` (fixes the 524 timeout on `/global-metric-bins`). | `dc0f2ff` |
| **6 Option A** | Test-window-only aggregation: training rows get bin set + `dropped_reason="pre_cutoff"`; all aggregators exclude them. UI label `TEST PERIOD · since YYYY-MM-DD` above the equity curve. Verified via 4 `train_test_check` tests (A/B/C/D all pass). | `3b03d03` |
| Cosmetic | Mode/Cutoff control alignment in the page header (label alignment + button sizing). | `573741d` |

## Train-test semantic (Option A)

Bins are **defined** by training-window data (`trade_date < cutoff`). Aggregations (per-bin avg-ret, win rate, Sharpe, equity curve, trade table, etc.) use **test-window rows only** (`trade_date >= cutoff`). Training rows still carry a bin assignment on the `RowAssignment` dataclass — tagged `dropped_reason="pre_cutoff"` — so a future side-by-side training/test view can iterate the same assignments without re-running the assigner.

For the user-facing question "does the training-defined threshold produce a useful out-of-sample signal?", this is the standard interpretation.

`cutoff > max(trade_date)` degenerates gracefully: every row is training, zero test rows, `assign_batch` returns `[]`, no crash. Covered by `train_test_check` Test D.

## Step 7 cleanup — DONE

All five checklist items resolved across Steps 7e–7k. Final state:

1. **`_sec_score_metrics` dual-signature** — done earlier in Step 7 (filter responsibility moved to caller via `filter_by_assignments`).
2. **Response-envelope forks in the secondary endpoints** — done in Step 7h via the `mode_envelope(spec, ...)` helper in `row_compute.py`. All four secondary endpoints emit a uniform 6-field envelope.
3. **Inline `/heatmap` single-ticker binning + `today_decile`** — `today_decile` migrated in Step 7i (now reads `pairs_decile[-1]` from the active Assigner, fixing a real bug where it could disagree with `trade_calendar[-1].decile`). `/heatmap` single-ticker stays on `np.percentile` edges — deliberate by-design exception (user-visible edge labels in the response).
4. **Dead legacy helpers** — done in Step 7j (`cdaa6f6`). Relocated 13 single-caller helpers from `oi_analysis.py` to `row_compute.py`. `_bucket_pairs` / `_bucket_pairs_per_ticker` / `_walk_forward_thresholds` stay in `oi_analysis.py` because they're still called by view-specific code (yearly_consistency, half-sample, threshold-drift). `oi_signals.py` now imports `_bin_for_value` from `row_compute`.
5. **Score Matrix train-test support** — done in Step 7k (`3bed28f`). Score Matrix's `_tt_bin_matrix` was using `searchsorted side='right'`, putting test values tied with interior training thresholds one or more bins too high. Consolidated onto a new shared primitive `train_test_bin_matrix_per_ticker` in `row_compute.py` with `side='left'` (matching `_compute_all_bins_train_test_fast` / `_bin_for_value` / `TrainTestAssigner`). Correctness verified by both a structural invariance argument and the targeted comparison in `scripts/tt_bin_tie_diff.py`. Affected feature list (pre-fix train-test Score Matrix values to distrust): `pct_up_days_20d` (87% wrong), `max_oi_strike_put` (33%, Δ up to -3), `max_oi_strike_call` (19%, Δ up to -3), `relative_strength_vs_spy_20d` (SPY-only artifact — SPY-vs-SPY is degenerate ~1.0), `pct_from_52w_high` (7%). 114 of 123 features unaffected; in-sample / walk-forward Score Matrix unaffected for all features. Re-run train-test Score Matrix after deploy to flush the contaminated rows. Score Matrix endpoints (`/score-matrix`, `/score-matrix/meta`, `/score-matrix/summary`) added to the regression matrix in `regression_check.py` — closes a blind-spot where the entire endpoint family was previously invisible to capture/diff.

## IC (Information Coefficient) tooling plan

Signal-stability tooling on top of rolling Spearman IC. Same step-cadence discipline as the row_compute refactor — one step per session with verification.

- **IC.1 — done (`b5c6407`)**. `app/routers/ic_compute.py` sibling module: three primitives (`rolling_ic_single_ticker`, `rolling_ic_cross_sectional`, `sign_stability_from_rolling`), mode-aware noise floor (`noise_floor_epsilon`), and `classified_rolling_ic` for per-window same/opposite/neutral labeling. Hand-computed verification in `scripts/ic_compute_check.py` (Tests A–E plus property checks).
- **IC.2 — done (`3e26016`)**. `/analyze` single-ticker `rolling_corr` field replaced by `rolling_ic` payload with per-window sign-class, mode-aware reference IC, ε, and sign-stability scalar. Pane upgraded to multi-segment colored line + reference dashed line + train_test cutoff marker. Regression diff against step7k confirmed bit-identity of every other `/analyze` field.
- **IC.3 — done (`c6a249b`)**. ALL-mode rolling IC populated via cross-sectional helper. Same payload shape as single-ticker, different ε (cross-sectional ε ~10× tighter). Frontend disambiguates with `[single-ticker]` vs `[cross-sectional]` subtitle tag. Regression diff confirmed only the predicted 4 diffs (ic_mode added in single-ticker, rolling_ic populated in ALL); all other endpoints PASS identical.
- **IC.4 — done (`05cd501` + hotfix `4aa8ff2`)**. New `/ic-batch` endpoint at `GET /api/oi-analysis/ic-batch?ticker=&outcome=&window=&cutoff_date=&refresh=`. DB-cached in table `ic_batch_cache` (mirror of `global_bins_cache` pattern). Compute runs off-event-loop via `asyncio.to_thread`. Returns `{metrics: [{name, long_run_ic, long_run_ic_abs, epsilon, n_windows, sign_stability, n_same, n_opposite, n_neutral, neutral_pct, suppressed, suppression_reason}, ...]}`. Mode encoded by `cutoff_date` param: set → train_test (reference IC from pre-cutoff windows only); unset → in_sample / walk_forward (full-history reference; same result for both since rolling IC is mode-independent except for the reference).

  **Fresh-fetch timing**: ~2–3 minutes for single-ticker on AAL × ~123 metrics × 1500 windows × per-window scipy.spearmanr. Cached subsequent reads are sub-second. ALL-mode fresh fetch is similar (DB fetch is the bottleneck; per-day Spearman on 80 tickers is faster than per-window Spearman on 252 pairs).

  **Visual verification done**: queried `ticker=AAL`, confirmed the result is consistent with the rolling-IC pane chart for AAL + `oi_weighted_call` (mostly-green chart matches `n_same=1260, n_opposite=0, n_neutral=339, sign_stability=1.0` in the batch result). The price-level metrics for AAL (`spot_co`, `oi_weighted_*`, `oi_above_spot_co`, etc.) consistently show n_opposite=0 across 1599 rolling windows — plausible given AAL's persistent multi-year downtrend; price-level-vs-forward-return is a structurally stable relationship for trending tickers within rolling windows.

- **IC.5 — next**. Frontend "Signal Stability" section on the Analyze page (between top selectors and "All Ticker Metric Bins"). Two universe-wide views, both driven by `/ic-batch`:
  - **IC stability leaderboard**: horizontal bar chart, one bar per metric, sorted by `sign_stability` desc. Bars color-coded by `long_run_ic_abs` (darker = stronger signal). Greyed-out bars for suppressed metrics. Header shows the active mode unambiguously (`Signal Stability — AAL (time-series)` vs `Signal Stability — ALL (cross-sectional)`) — bound to the ticker selector and updates in lockstep.
  - **Strength-vs-stability scatter**: every metric a dot. X = `long_run_ic_abs`. Y = `sign_stability`. Quadrants carry meaning: top-right = strong + persistent (priority); top-left = weak + persistent; bottom-right = strong + unstable ("one regime" trap); bottom-left = noise.
  - **Click-to-load**: clicking a bar or dot flips the `metric` selector and triggers `/analyze`, dropping the user from survey mode into deep-dive on that metric.
  - Implementation notes: new section in `templates/oi_analysis.html` before the existing "All Ticker Metric Bins" pane. New render functions in `static/js/oi_analysis.js`. Both views read `data.ic_batch.metrics`; the Alpine fetch hook calls `/ic-batch` on page load and on `setPageMode()` (mode change recomputes via the cache).

- **IC.6 — pending after IC.5**. Add `/ic-batch` (3 modes — default cached, train_test with a fixed cutoff, refresh=true to force fresh) and the updated `/analyze` rolling_ic payload variants to the regression capture matrix. Capture a new baseline tag. Verify nothing else regresses.

- **IC.7 — placeholder (don't design or build yet, just tracked)**. Per-ticker decomposition of cross-sectional IC for a selected metric: a bubble scatter diagnosing whether a metric's edge is broad across the universe or concentrated in a few names. Measured against the *cross-sectional* noise floor, NOT single-ticker IC. Diagnosis tool, not a ticker-selection tool. To be specified in detail after IC.5 ships, since it drills into the leaderboard.

## Verification harness (`scripts/regression_check.py`)

Three modes:

- `capture --tag <name>` — Hits 14+ endpoints against a running dashboard (`--base http://100.76.94.99:8000/api/oi-analysis` for the VPS), writes JSON snapshots into `regression_snapshots/<tag>/`. Has retry-with-backoff for transient VPS hiccups (uvicorn worker respawns from OOM on the 60MB ALL-mode `/analyze` responses). Matrix includes /analyze ×4, /global-metric-bins ×2, /heatmap ×3, secondary-load + secondary-corr-bins chain ×2, portfolios list + aggregate ×2.
- `diff --before X --after Y` — Recursive structural diff with `math.isclose(rel_tol=1e-9)`. Lists of dicts keyed by `(ticker, trade_date)` compare ordering-insensitive.
- `train_test_check` — Four hand-verified train-test correctness tests (A: 10-row small case bins, B: cutoff==max property, C: `pre_cutoff` tagging, D: empty-test-set degeneracy). All four currently PASS.

**Reference baseline** for in-sample/walk-forward bit-equivalence: `regression_snapshots/step3-baseline/` (captured against the VPS at commit `e866848`, post-Step-2 with all the bit-equivalence follow-ups). Every subsequent migration step (3, 4, 5, 5.5) was diffed against this and came back clean except for environmental drift (`cached_at` timestamps on `/global-metric-bins`, FP drift on `/secondary-load` lift values — both 2a-accepted).

Step 7j (`cdaa6f6`) was also diffed against `step3-baseline` (snapshot tag `step7j`). 246 diffs, all explainable as: row-count drift from ~3 weeks of additional ingested `daily_features` rows (~190 diffs), pre-existing Step 7f `cutoff_date` envelope additions (~6), pre-existing Step 7h `mode_envelope` unification (~6), exit prices materializing on trades whose `exit_date` fell between baseline and Step 7j capture (~10), `cached_at` timestamps (2), heatmap edge labels shifting with newer data (~10), two new heatmap matrix variants from Step 5.5. No diff indicates Step 7j broke Assigner behavior — all numeric shifts are row-count-proportional, consistent with structural relocation.

Step 7k (`3bed28f` + `507baa7` + `a16df70`) snapshot tag `step7k` was diffed against `step7j`. 14 of 17 files PASS identical (every Assigner-routed endpoint); 9 score-matrix files baseline themselves (new matrix coverage); 3 files with non-algorithmic diffs (`secondary-corr-bins__wf0` got `cache_miss` due to a uvicorn worker restart between captures; the two `secondary-load` files have fourth-decimal `lift` drifts from ~24 hours of additional `daily_features` data). No collateral damage from the Score Matrix consolidation. `step7k` is the new clean baseline going forward.

## Operational

**VPS**: at Tailscale `100.76.94.99:8000`, from `/spx_analysis_dashboard` via `python run.py`. SSH `root@100.76.94.99` works with key auth from the original machine. The dashboard uses asyncpg+FastAPI+uvicorn (`--reload` mode). uvicorn workers can OOM on the heavy `/analyze` ALL-mode responses (60MB+) and respawn — the regression harness handles this with retry-with-backoff.

**Deploy protocol per step**: commit + push on dev machine → user SSHes to VPS, `git pull`, restart dashboard → user warms cache (loads the OI Analysis page once) → user reports ready → I run capture + diff → user does manual click-through where the plan calls for it (e.g. Step 2's 12-view hard stop).

**Working cadence (in memory)**: one migration step per session, manual review between each, no chaining. See `C:\Users\kroko\.claude\projects\C--Personal-Data-spx-analysis-dashboard\memory\feedback_step_cadence.md` for the user's explicit phrasing.

## Critical files

| File | Role |
|------|------|
| `app/routers/row_compute.py` | Refactor destination — Protocol, registry, 3 Assigner classes, 4 dispatch functions, `PortfolioVectorBuilder`, `make_spec`, `dropped_count_for_mode`, `_validate_assignments`. Also hosts the relocated binning helpers from Step 7j and `train_test_bin_matrix_per_ticker` (the Step-7k shared primitive Score Matrix now uses). |
| `app/routers/ic_compute.py` | **NEW** sibling module added in IC.1. Three primitives: `rolling_ic_single_ticker`, `rolling_ic_cross_sectional`, `sign_stability_from_rolling`. Plus `noise_floor_epsilon` (mode-aware ε), `classified_rolling_ic` (single source of truth for per-window sign labels), and `IcPoint` / `SignStability` dataclasses. Date-indexed-series output — doesn't unify cleanly with row_compute's per-row contract; intentional parallel module. |
| `app/routers/oi_analysis.py` | Hosts the migrated endpoints. New in IC.2/IC.3: `/analyze` emits `rolling_ic` payload (was `rolling_corr`). New in IC.4: `/ic-batch` endpoint + `ic_batch_cache` table + `_compute_ic_batch_sync`. `_walk_forward_thresholds`, `_bucket_pairs`, `_bucket_pairs_per_ticker` still live here (used by yearly_consistency, half-sample, threshold-drift — view-specific code, not Assigner pipeline). |
| `app/routers/oi_portfolios.py` | `/portfolios/{pid}/aggregate` — uses `PortfolioVectorBuilder`. |
| `app/routers/oi_signals.py` | Imports `_bin_for_value` from `app.routers.row_compute` (moved in Step 7j). |
| `research/batch_score.py` | Score Matrix scanner. `_tt_bin_matrix` is now a thin delegator to `row_compute.train_test_bin_matrix_per_ticker` (Step 7k). |
| `static/js/oi_analysis.js` | Frontend. `pageMode` state, `setPageMode(m)` cascade. IC.2/IC.3 added `_renderRollingCorr` rewrite (multi-segment colored line + reference dashed line + cutoff marker) and `rollingIcSubtitle()`. Cache version currently **v=86** (controlled in `templates/oi_analysis.html`'s `<script src="...?v=NN">`). |
| `templates/oi_analysis.html` | 3-way Mode toggle + Cutoff date input. Rolling-IC pane subtitle div. Cache version v=86. |
| `scripts/regression_check.py` | Verification harness. Runs against the live VPS. Matrix now includes 9 score-matrix captures (added in Step 7k) — total 26 files per capture. |
| `scripts/ic_compute_check.py` | **NEW** hand-computed verification for `ic_compute` primitives. 8 tests (ε derivation + A/B/C/D/E + 2 properties). Runs without DB. |
| `scripts/tt_bin_tie_diff.py` | **NEW** one-off targeted comparison: pulls real `daily_features` and computes train-test bins under both `side='right'` (pre-Step-7k) and `side='left'` (post). Runs on the VPS where OI_DATABASE_URL is local. Used to identify the 5 contaminated features. |
| `C:\Users\kroko\.claude\plans\stateful-tumbling-dragon.md` | The original refactor plan (Step 7 cleanup checklist). |
| `regression_snapshots/` | **Gitignored**. Snapshot tags `step7k`, `step_ic2`, `step_ic3` (and others) only exist on the machine that captured them. Re-capture as needed on the new machine. |

## Picking up — concrete first steps on the new machine

1. `cd C:\Personal\Data\spx_analysis_dashboard`
2. `git pull` — latest should be `4aa8ff2` (IC.4 hotfix — renamed `ic_batch_cache.window` → `window_size` to dodge the PostgreSQL reserved-word collision).
3. If `.venv` is fresh: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt` (numpy, scipy, asyncpg, fastapi, etc. — `requirements.txt` is in the repo).
4. Smoke-test both verification suites:
   - `python scripts\regression_check.py train_test_check` → expect A/B/C/D PASS (TrainTestAssigner property tests; verifies the row_compute refactor still works).
   - `python scripts\ic_compute_check.py` → expect ε derivation + Tests A/B/C/D/E + properties all PASS (verifies the IC tooling primitives).
5. **regression_snapshots/ is gitignored**. The `step7k`, `step_ic2`, `step_ic3` baseline tags from prior sessions do NOT carry over via git. To run any future diff against a baseline, capture a new tag on the new machine first (typically against the running VPS): `python scripts\regression_check.py capture --tag <name> --base http://100.76.94.99:8000/api/oi-analysis`. The full matrix takes ~3 minutes and produces 25–26 files. There IS no out-of-band sync for snapshot dirs.
6. **VPS state**: at last check the VPS is on `4aa8ff2` (the user deployed before the session ended). The `ic_batch_cache` table will be created on first /ic-batch hit if it doesn't exist already. The `oi_score_matrix` table on the VPS still contains pre-7k train-test scores for the 5 contaminated features — those should be re-scanned via a new train-test Score Matrix run before relying on train-test Score Matrix values for those features.
7. **Next work item: IC.5** (frontend leaderboard + scatter). Detailed plan in the IC section below.

## Recent commits (for context)

```
4aa8ff2  IC.4 hotfix: rename ic_batch_cache.window column (PG reserved word)
05cd501  IC.4: /ic-batch endpoint + log IC tooling plan (incl. IC.7 placeholder)
c6a249b  IC.3: enable rolling IC + sign-stability in ALL mode (cross-sectional)
3e26016  IC.2: rolling IC + sign-stability in /analyze single-ticker pane
b5c6407  IC.1: ic_compute.py sibling module + hand-computed verification
a325482  Close out Step 7: all five cleanup items resolved (7e–7k)
3bed28f  Step 7k: consolidate Score Matrix train-test bins onto shared primitive
cdaa6f6  Step 7i + 7j: migrate today_decile, relocate 13 single-caller helpers
66550f3  Step 7h: unify mode envelope across 4 secondary endpoints
fb11fbb  Step 7g: fix global_bins_cache DB read (jsonb returned as str)
66fedc7  Step 7f: persist train-test score matrix per cutoff
6653a3c  Step 7e: NaN-safe score matrix endpoints
fb7d62a  Step 7d: thread the batch score's CPU-heavy per-ticker work
3b03d03  Step 6 Option A: test-window-only aggregation for train_test
e782401  Step 6: register train_test as the third method end-to-end
ce6fc4a  Step 5.5: route /heatmap ALL-mode through Assigner
080cd99  Step 5: route /portfolios/{pid}/aggregate through row_compute
7731f61  Step 4: route 4 secondary endpoints through row_compute
ee92d72  Step 3: route /global-metric-bins through row_compute
4c70892  Step 2: route /analyze through row_compute layer
9ffc284  Step 1: row_compute module + regression harness (purely additive)
```

## User-context worth knowing

- **124 tickers** in the dataset (a few without 2019 data — affects warmup math for walk-forward).
- Default outcome dropdown is `ret_5d_fwd_oc`. The init() force-selects this via DOM manipulation because Alpine reactivity wasn't reliable for default-selection across browser form-state cache.
- User prefers terse responses, explains the **why** rather than the **what**, reads the diff themselves.
- For exploratory questions, give a recommendation with the main tradeoff and let them redirect.
- User pushes back on descoping ("don't leave follow-up items I'll forget") — when in doubt, finish the item now or add it to the plan's Step 7 cleanup checklist.

## Auto-memory note

`C:\Users\kroko\.claude\projects\C--Personal-Data-spx-analysis-dashboard\memory\` holds two memory files:
- `project_deployment.md` — VPS deployment summary
- `feedback_step_cadence.md` — one step per session, manual review between

Memory is per-user/per-machine. Copy the directory across or rely on this doc.

---

**TL;DR for the next session**: row_compute refactor done (Steps 1–7). IC tooling at IC.4 done and visually verified. **Next work item: IC.5 — Signal Stability section on the Analyze page (leaderboard + scatter, both driven by `/ic-batch`, both wired to the ticker selector, click-to-load on bars/dots).** Specifics in the IC plan section above. Snapshot baselines don't carry over via git — re-capture as needed.
