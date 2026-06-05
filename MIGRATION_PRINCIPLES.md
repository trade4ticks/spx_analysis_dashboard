# Bin-Migration Principles

Reference these when migrating any view to read stored bins (`is_bins`,
`wf_bins`, `tt_thresholds`). Established through Groups 1–4 (in-sample
heatmap, `/metric-bins`, `/analyze`, `/analyze-bundle`, secondary
endpoints); they apply to WF (Group 7) and TT (Group 8) by reference,
not by re-derivation.

## The four rules

### 1. No per-ticker thinning
The only row filter is `stored_bin > 0`, which already excludes the
null-metric sentinel set by the bin-table build. No `n_t < N` exclusion,
no sentinel rewrites to bin 0, no per-ticker count gates anywhere in
dashboard code. Sparsity is the user's data problem, not the
dashboard's defensive layer.
- Pattern in code: `WHERE ib.bin20_<metric> > 0` at the SQL JOIN.
- Counter-example removed in Part 1: the `n_t < 10` ticker drop +
  `n_t < 20` → `decile20 = 0` sentinel in `/analyze`'s IS+ALL branch
  (now: every row carries its real stored bin).

### 2. No re-ranking
A row's bin is a property of the row, not of the filtered subset it's
being shown in. Read the stored bin verbatim. Never recompute a
per-ticker rank against a primary-filtered or otherwise narrowed set.
- Pattern in code: `bin20_lookup[(ticker, date)] = r['bin_20']`; the
  display loop reads from the lookup, doesn't re-rank.
- Counter-example: the pre-v9 `_compute_bins_for_metric` re-rank on
  filtered subset is preserved in `row_compute.py` for legacy WF/TT
  paths only and falls out when those migrate.

### 3. One shared 20 → N collapse formula
Display granularity comes from the stored 20-bin via:
```
bn = min(((b20 - 1) * N) // 20, N - 1)   # 0-indexed bucket
b1 = min(((b20 - 1) * N) // 20 + 1, N)   # 1-indexed bin
```
For `N ∈ {5, 10, 20}` (the granularities the UI exposes — all divisors
of 20), this is a direct per-ticker rank. For non-divisor N it shifts;
the UI doesn't expose those. Same formula in every endpoint — search
for `(b20 - 1) * N // 20` to find every site.

### 4. Cache-salt bump on every bin-math change
Every persistent cache that stores bin assignments salts its
`cache_key` with a schema-version constant. Any change to bin
assignment math OR response shape **must** bump that constant in the
same commit. Without the bump, deployed code computes fresh bins but
on-disk payloads keep serving the old ones until manual truncate —
this is the bug that caused the heatmap-vs-secondary 31-row gap.

Current constants and what they salt:

| Constant | Cache | Key prefix |
|---|---|---|
| `_ANALYZE_PRIMARY_SCHEMA_VERSION` | `analyze_primary_cache` | `ap:vN:` |
| `_ANALYZE_BUNDLE_SCHEMA_VERSION` | `analyze_cache_slim/trade_meta/outcome` | `ab:vN:` |
| `_GLOBAL_BINS_SCHEMA_VERSION` | `global_bins_cache` | `gb:vN:` |

Each `_ensure_*_table()` runs a `DELETE ... WHERE cache_key NOT LIKE
'<prefix>%'` on startup, so a bump self-cleans the table on the next
deploy.

If you add a new salt-keyed cache, mirror the pattern: constant + key
prefix + ensure-table sweep + a discipline-comment block referencing
this file.

## WF-specific caveat (Group 7)

Walk-forward sparsity is **structural**, not a data-quality artifact.
The first 252 trading days (the warm-up window) genuinely have fewer
prior observations than the rest of the series — that's how
walk-forward computation works. So "no thinning" in WF mode doesn't
mean "emit a bin for day 1 against zero history"; it means **decide
explicitly what to do at the warm-up boundary** before writing the
read-time code, then encode that decision in the build script
(`wf_bins`), not in the reader.

Pre-migration questions to answer for WF:
1. Where exactly does `wf_bins.bin20_<metric>` start emitting non-zero
   bins for each ticker? (Build decision.)
2. Do dashboard views render the pre-warm-up gap as empty cells,
   omitted rows, or annotated "warm-up" rows? (UX decision.)
3. Does `/analyze`'s `trade_calendar.decile20` distinguish "warm-up,
   no bin yet" from "real bin = 0 sentinel"? (Payload decision.)

The IS rule (no thinning, `bin > 0` is the only filter) only applies
once those three questions are answered such that `wf_bins.bin20 > 0`
means "real bin assignment from sufficient history" and bin 0 means
"warm-up". Don't copy the IS migration mechanically.

## TT-specific caveat (Group 8)

Train-test uses frozen thresholds from `tt_thresholds`. Two natural
sentinels collide there: (a) row predates the train/test cutoff, (b)
metric was null at threshold-derivation time. The cache salt must
include the cutoff date (it already does in `_analyze_primary_cache_key`
and the bundle key), and the build script needs to decide which
sentinel encodes which case so the read-side filter `bin > 0` can stay
on one principle.

## Verification standard

Sign-off is **node-for-node tie-out across all on-screen surfaces that
share a row population**, not a curl tie-out against pgAdmin. Curl
against pgAdmin only proves a query is self-consistent. The 31-row gap
in Group 4 was invisible to curl tie-outs of either endpoint
individually and only surfaced when the heatmap (which read is_bins
live) was compared against the secondary view (which read
`trade_calendar.decile20` from a stale cache). The on-screen
comparison is the bar.
