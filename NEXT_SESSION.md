# SPX Dashboard — Session Handoff (row_compute refactor, Steps 1–6 done)

This doc is a self-contained briefing for picking up the row_compute refactor on another machine. Read top to bottom.

## TL;DR

A multi-step refactor extracts the binning computation out of the FastAPI endpoint bodies into a single swappable layer (`app/routers/row_compute.py`). The page-wide `Mode: In-sample | Walk-fwd | Train-test` toggle now drives every binning analysis through one dispatch surface. Steps 1–6 are done and shipped. Step 7 (cleanup of residual forks + dead helpers) is pending.

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

## What's left — Step 7 cleanup

The full checklist is in `C:\Users\kroko\.claude\plans\stateful-tumbling-dragon.md` under "Step 7 cleanup checklist". Summary:

1. **`_sec_score_metrics` dual-signature** — currently accepts two distinct call shapes via the truthiness of `filtered_dates`. Used only by `/secondary-scan` post-Step-4. Refactor to a single shape `(rows, outcome, features, is_all)` with the filtering done by the caller.
2. **Response-envelope forks in the secondary endpoints** — `/secondary-corr-bins`, `/secondary-correlation`, `/secondary-detail`, `/secondary-scan` still build response dicts with `if spec.kind != "in_sample"` branches because non-in-sample modes include extra metadata (`warmup` / `cutoff_date` / `dropped_warmup_n` / `start_date`). Unify into one envelope builder that consults the spec.
3. **Inline `/heatmap` single-ticker binning and `today_decile`** — `/heatmap` single-ticker uses `np.percentile` edges (deliberately different algorithm; user-visible edge labels). `today_decile` at `oi_analysis.py:705` is a one-off inline formula. Decide whether to keep or fold.
4. **Dead legacy helpers** — `_walk_forward_primary_filter`, `_compute_walk_forward_bin_stats`, `_walk_forward_membership`, `_bucket_pairs_per_ticker`, etc. became single-caller (each Assigner method or dispatch function that delegates to them). Either inline into the Assigner methods or delete the helpers and have the Assigners do the math directly.
5. **Score Matrix train-test support** — `/score-matrix-batch` accepts only `walk_forward: bool` today. In train-test pageMode the frontend sends `walk_forward=false`, so Score Matrix runs in_sample. Add `cutoff_date` and route through `TrainTestAssigner` + `_compute_all_bins_train_test_fast` (or similar).

## Verification harness (`scripts/regression_check.py`)

Three modes:

- `capture --tag <name>` — Hits 14+ endpoints against a running dashboard (`--base http://100.76.94.99:8000/api/oi-analysis` for the VPS), writes JSON snapshots into `regression_snapshots/<tag>/`. Has retry-with-backoff for transient VPS hiccups (uvicorn worker respawns from OOM on the 60MB ALL-mode `/analyze` responses). Matrix includes /analyze ×4, /global-metric-bins ×2, /heatmap ×3, secondary-load + secondary-corr-bins chain ×2, portfolios list + aggregate ×2.
- `diff --before X --after Y` — Recursive structural diff with `math.isclose(rel_tol=1e-9)`. Lists of dicts keyed by `(ticker, trade_date)` compare ordering-insensitive.
- `train_test_check` — Four hand-verified train-test correctness tests (A: 10-row small case bins, B: cutoff==max property, C: `pre_cutoff` tagging, D: empty-test-set degeneracy). All four currently PASS.

**Reference baseline** for in-sample/walk-forward bit-equivalence: `regression_snapshots/step3-baseline/` (captured against the VPS at commit `e866848`, post-Step-2 with all the bit-equivalence follow-ups). Every subsequent migration step (3, 4, 5, 5.5) was diffed against this and came back clean except for environmental drift (`cached_at` timestamps on `/global-metric-bins`, FP drift on `/secondary-load` lift values — both 2a-accepted).

## Operational

**VPS**: at Tailscale `100.76.94.99:8000`, from `/spx_analysis_dashboard` via `python run.py`. SSH `root@100.76.94.99` works with key auth from the original machine. The dashboard uses asyncpg+FastAPI+uvicorn (`--reload` mode). uvicorn workers can OOM on the heavy `/analyze` ALL-mode responses (60MB+) and respawn — the regression harness handles this with retry-with-backoff.

**Deploy protocol per step**: commit + push on dev machine → user SSHes to VPS, `git pull`, restart dashboard → user warms cache (loads the OI Analysis page once) → user reports ready → I run capture + diff → user does manual click-through where the plan calls for it (e.g. Step 2's 12-view hard stop).

**Working cadence (in memory)**: one migration step per session, manual review between each, no chaining. See `C:\Users\kroko\.claude\projects\C--Personal-Data-spx-analysis-dashboard\memory\feedback_step_cadence.md` for the user's explicit phrasing.

## Critical files

| File | Role |
|------|------|
| `app/routers/row_compute.py` | The refactor's destination — Protocol, registry, 3 Assigner classes, 4 dispatch functions, `PortfolioVectorBuilder`, `make_spec`, `dropped_count_for_mode`, `_validate_assignments`. |
| `app/routers/oi_analysis.py` | Hosts the migrated endpoints + the legacy helpers that the Assigners delegate to. `_compute_all_bins_walk_forward` and `_compute_all_bins_train_test_fast` live here (numpy-vectorized batch helpers). |
| `app/routers/oi_portfolios.py` | `/portfolios/{pid}/aggregate` — uses `PortfolioVectorBuilder`. |
| `static/js/oi_analysis.js` | Frontend. `pageMode` state, `setPageMode(m)` cascade, all fetch sites send `walk_forward` / `cutoff_date` per mode. Cache version controlled by the `<script src="...?v=NN">` line in `templates/oi_analysis.html` — currently **v=81**. |
| `templates/oi_analysis.html` | 3-way Mode toggle + Cutoff date input in the page header. Subtitle templates for walk_forward (orange) and train_test (mauve). Explicit `TEST PERIOD · since YYYY-MM-DD` label above the Equity Curve canvas in train-test mode. |
| `scripts/regression_check.py` | Verification harness. Runs against the live VPS. |
| `C:\Users\kroko\.claude\plans\stateful-tumbling-dragon.md` | The refactor plan. Step 7 cleanup checklist lives there. |
| `regression_snapshots/step3-baseline/` | Reference baseline for in-sample/walk-forward bit-equivalence. Gitignored. ~204 MB. |

## Picking up — concrete first steps on the new machine

1. `cd C:\Personal\Data\spx_analysis_dashboard`
2. `git pull` — latest should be `573741d` (cosmetic alignment fix).
3. If `.venv` is fresh: `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt` — note `requirements.txt` now explicitly includes `numpy>=1.26.0` (Step 1 added it because the implicit transitive pin from scipy was insufficient).
4. Verify `regression_check.py train_test_check` passes: `.venv\Scripts\python.exe scripts\regression_check.py train_test_check`. Expected: A/B/C/D all PASS.
5. The user is currently testing Step 6 Option A in the browser (test-window-only aggregation). If they greenlight, **Step 7 cleanup is next**. Pick one item from the cleanup checklist (`_sec_score_metrics` dual-signature is the most-discussed; envelope unification is the most-pervasive).

## Recent commits (for context)

```
573741d Mode/Cutoff control alignment + size to match Analyze button
3b03d03 Step 6 Option A: test-window-only aggregation for train_test
dc0f2ff Step 6 hotfix #2: vectorize TrainTestAssigner.assign_batch (524 fix)
fd93d0c Step 6 hotfix: TrainTestAssigner preserves trade_date type
e782401 Step 6: register train_test as the third method end-to-end
025ab50 Step 5.5 continuation: heatmap respects pageMode end-to-end
ce6fc4a Step 5.5: route /heatmap ALL-mode through Assigner
080cd99 Step 5: route /portfolios/{pid}/aggregate through row_compute
7731f61 Step 4: route 4 secondary endpoints through row_compute
ee92d72 Step 3: route /global-metric-bins through row_compute
19bfe2f Step 2 1a (cont.): match legacy within-date pair order + retry harness
8f5897b Step 2 1a: match legacy bucket iteration order
e866848 Step 2 followup: preserve legacy decile20=0 for sparse-ticker rows
4c70892 Step 2: route /analyze through row_compute layer
9ffc284 Step 1: row_compute module + regression harness (purely additive)
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

**TL;DR for the next session**: Steps 1–6 of the row_compute refactor are done. The page-wide Mode toggle drives every binning surface (Score Matrix excepted, flagged for Step 7). Train-test mode uses test-window-only aggregations per Option A. Currently waiting on the user's manual verification of Step 6 Option A. **Next work item: pick something from the Step 7 cleanup checklist in the plan file.**
