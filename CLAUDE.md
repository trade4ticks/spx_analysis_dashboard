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
**half-open `[entry, exit)`** — overnight capital, so a position is counted on the
sessions it is held through and not on the one it closes on (changed 2026-09-25; see the
Backtest Portfolio section for why, and note it lowers peak concurrency and therefore Avg
Annual Return %). Every session in the span is a point, so a stretch with
nothing open is a run of zeros. One stepped line; the right axis is the left × capital,
pinned to the same range, not a second trace. Still-open MesoSim positions are excluded by
the parser and so contribute nothing (gate-checked against the v3.1 fixture). Trades
opening or closing off an exchange session (a weekend or holiday) are counted from the
next session and reported under the chart. 2026-04-08 is no longer an example of this —
it IS a session; see the exchange-calendar section.

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

**`replace(qty=…, filled=…)`: `qty` is the order's TOTAL.** Brokers mean different
things by the number on the wire, so the caller sends the order's own two figures and
the adapter converts: DAS MODIFIES the resting order and sends the total (its own
record's, where it has one); Schwab cancels and places a new order and sends
`qty − filled`. The pane sending the remaining shrank partially filled DAS orders on
every nudge — 5 → 4 → 3 → 2 with no fills (2026-09-22) — because each reply's
remaining became the next request's total.

**An order with nothing left is not working, whatever its status says**
(`das.is_working`). DAS returns `Partial` after a cancel of a partly filled order
(`… 2 0 1 … Partial`: two ordered, zero left, one cancelled): the status records what
HAPPENED, not whether anything rests, so it outlives the order. Taking it as live kept
a ghost on the ladder that a nudge could not move and a cancel answered "order not
open". Exception: `HOLD`/`SENDING` with zero left stay working — nothing rests YET,
and hiding an order about to be live (with its cancel) is the direction this file
never takes. A layout without `lvqty` falls back to the status.

**Only a validated DUMP HEADER touches the DAS snapshot, and only an END marker
sets it.** Twice a line from the `#Order` family blinded the pane: `#OrderServer`
(2026-09-17, a prefix test) and again on 2026-09-21 despite exact-token matching —
because an `#Order` line carrying an ORDER has the same head as the header row and
cannot be told apart by the head at all. The rule is now structural, not a list of
exceptions: a dump opens only on the documented field-name row (`is_dump_header`:
first field `ID`/`SYMB`, no digits anywhere), the snapshot flag is NEVER cleared by
a header (only set by `#OrderEnd`/`#POSEND`, only reset on connect), and a dump that
never gets its END is abandoned after `DAS_DUMP_TIMEOUT_S` so pushes go back to the
live record. Anything else in the family is counted as `#ORDER(info)` in
`LINK.unhandled` → `health().socket.unhandled`. Symptoms when this breaks: every read
"order list has not arrived yet (6s)" (that 6s is `DAS_SNAPSHOT_S`, the wait, not an
age) plus CANCEL/REPLACE "order not open", because the process is targeting orders
DAS has already closed.

**DAS dispatch matches the EXACT first token**, never a prefix. `#OrderServer`
(a routine status push) shares its prefix with `#Order`, so a prefix test read it as
the header of a fresh order snapshot: it cleared `order_snapshot` — so `_ready`
refused every read for the session, the pane stayed empty, and `/broker/state`
answered "its order list has not arrived yet (6s)" forever (that 6s is
`DAS_SNAPSHOT_S`, the configured wait, not an age) — and it opened a staging
buffer no `#OrderEnd` ever closed, so orders pushed afterwards went where nothing
reads. Orders reached DAS and worked; none ever drew. Ordering the checks fixes
only the END markers; the next `#Order*` line breaks it again. Unrecognised heads
are counted in `LINK.unhandled` and surface in `health().socket.unhandled`
(`#SLOrder` is the standing example). The real login banner is captured verbatim in
`check_das.LOGIN_BANNER` — an account with NO orders sends only headers and END
markers, and that is a complete snapshot. The first `LIVE_DAS_LOG_LINES` (60) lines
of every connect are logged verbatim at INFO, the first line of each new kind once,
and — while `LIVE_DAS_LOG_ORDERS` is on (default) — EVERY `%ORDER` and `%OrderAct`,
because first-of-each-kind hides the stream that follows one order from sent to
filled. The `DAS placed` line prints the WIRE price (`fmt_price`), not the float it
came from: 1158.6000000000001 in the journal reads as a sub-penny limit nobody could
have placed, while the wire carried 1158.60.

**The DAS route list is the MONTAGE** (`LIVE_DAS_ROUTES`, 45 base names with the
montage's L/M suffix stripped), not `GET RouteStatus`: RouteStatus reports everything
the login can see — options, short-locate, test and PRO routes Cobra does not expose —
and none of those is somewhere to send an equity order. RouteStatus only MARKS each
montage entry in `health()["routing"]["states"]`: `enabled` / `disabled` / `unconfirmed`
(never mentioned, or nothing heard yet — PSMT is the standing case). Disabled entries
stay on the list, greyed, so the dropdown does not change shape between pre-market and
the session, and all three states stay selectable: DAS decides what it accepts, and a
stale snapshot here must not block a live venue. `check_das.case_route_list` plants a
reply with stray non-montage routes and a missing PSMT.

Adapter internals are now `schwab.*` — the checks and probes monkeypatch
`schwab._acall` / `schwab._account_hash`, not `broker.*`. Rate limits belong to the
adapter (a broker-API fact); the rule they serve — getting flat must never be
refused for quota — stays in the façade's `priority` flags.

### DAS Trader, the second adapter (2026-09-20)

`live/brokers/das.py`, selected with `LIVE_BROKER=das`. Schwab is untouched
and still the default. **It is a socket, not an API**, and that is the whole
difference: one persistent TCP connection to DAS Trader Pro on the Windows
machine (Tailscale address in `LIVE_DAS_HOST`, port 9910), plain-text lines,
CRLF. DAS must be running and logged in for the socket to exist.

**State is PUSHED** (`%ORDER`, `%OrderAct`, `%TRADE`, `%POS`), so a read is a
cache lookup, not a call — microseconds against Schwab's ~850 ms. Nothing
polls, which means no request's success stands in for freshness: `as_of` is
**when the socket was last proven alive**, and an `ECHO` heartbeat every
`LIVE_DAS_HEARTBEAT_S` (3s) produces that proof on a quiet name. A snapshot
(`#POS`/`#POSEND`, `#Order`/`#OrderEnd`) REPLACES the cache; the end markers
are tested before the start markers, because `#POSEND` starts with `#POS`.

**The token is a real client order id**, so `reconcile` is overridden and
matches on it exactly — `ambiguous` is unreachable, unlike Schwab's shape
match. The token→placement map is this process's; if the service restarted
between the placement and the reconcile it falls back to
`base.match_placement` and says `matched_on: "shape"`. Tokens are C ints,
minted high in the positive half so they cannot collide with the montage's.

**The acknowledgement is claimed before the command is written.** Over a
local socket the `%ORDER` can arrive inside the `await` that sends the
NEWORDER; registering after would lose it and report a resting order as
UNKNOWN. Gate-covered by pushing the reply synchronously from the write.

**`TimeOut` and `Send_Rej` are `BrokerIndeterminate`** and never retried; a
`%ORDER` status of `Rejected`, and `CancelRej`/`ReplaceRej`, are determinate
`BrokerError`. Working statuses: `Hold`, `Sending`, `Accepted`, `Partial`,
`Triggered`; terminal: `Closed`, `Canceled`, `Rejected`, `Executed`; an
unrecognised status resolves to **working**, as the interface requires.

**A quiet or dropped link still answers**, from the cache, with `as_of`
frozen so the age climbs — DAS Trader is on screen showing the same orders,
so blocking the pane would interrupt more than it protects. The one refusal
is a read taken before the first snapshot has EVER arrived: an empty list
there is a fabrication, not a stale answer.

**Deliberately absent:** no market data (`SB`, Level 1/2, time & sales,
charts) — the tape is Polygon's and there is no depth of book in this API;
`check_das.py` fails if a subscription command appears. `%POS Unrealized` is
not used (the manual says it is a snapshot from when the position was sent);
`day_pl` carries `Realized` and the pane computes open P&L from the tape.

**Routing is on the interface** (`place`/`replace` take `route`) because
venue control is the reason for the move to Cobra. `None` means the
adapter's default; Schwab accepts it, ignores it and reports
`routing.supported = False` from `health()`, so the page draws no control
there. The venue list comes from DAS itself (`GET RouteStatus`). DAS's
`REPLACE` carries no route, so a reprice onto a different venue is
**refused**, not silently ignored — changing venue means cancel and place.
`das.build_neworder` already builds `PostOnly`, `NotRouteOut`, `TIF`,
`Display`, `Minume` and `Pref`; wiring any of them up is interface, façade,
endpoint and control, and the protocol half is written and gate-covered.

Gate: `scripts/check_das.py`, 19 cases against a fake socket (no network, no
DAS). Runs on the VPS too — pure Python.

### The marketable-click guard: only a definite cross asks (2026-09-20)

`askFirst` used to raise the confirmation whenever `isMarketable` returned
null — no current NBBO — on the reasoning that unknown is not safe. Removed:
the names traded here are deliberately quiet, so a thirty-second-old quote
is the setup rather than a warning sign, and the banner fired on nearly
every click in exactly those names. A confirmation that always fires is one
that gets clicked through without being read. Live trade prints arrive
continuously, so a dead feed has no prints at all, and lagging quotes beside
arriving prints still say where the market is.

**What stays:** the warning for a price that crosses a CURRENT touch, and
`isMarketable` answering null rather than false for a stale or missing quote
(a definite "this will rest" off an unseen touch is a wrong warning, and the
wrong one is what moves money). The ladder hatching and the drag label now
mark only what is known to cross, for the same reason.

## Backtest Portfolio (`/backtest-portfolio`, in progress — 2026-09-24)

Several SAVED strategies combined: per-strategy filters, qty and capital,
and how the combination performs. Spec'd from the old Dash app
(`Options-Backtest-Dashboard/pages/portfolio.py`, `utils/portfolio_calcs.py`)
read as a specification, not a template.

**Phases:** P1 scaffold + load — done; P2 filters, allocation, summary
table with TOTAL row — done; P3 equity/drawdown/capital deployed/monthly
P&L — done; P4 correlation — done; P5 saved
profiles — done; P6 distribution, overlap, rolling risk — done; P7 docs.

**One core, two pages.** `static/js/backtest_core.js` holds every shared
calculation — `obApplyFilters`, `obStats`, `obEquity`, `obExtraStats`,
`obConcurrency`, `obSharpe`, `obDeployedSeries` and the formatters — moved
out of `oo_backtest.js` and read by both. The portfolio page defines no
statistic of its own; the gate asserts that in both directions (the page
calls them, the page does not define them, the core does, the OO page no
longer does). **The file has no `'use strict'` on purpose**: the gates run
the shipped functions by `eval`ing the core together with a page's bundle in
node, and a strict eval keeps its declarations to itself — every driver would
see `obStats is not defined`. `obNull` is a `const`, so the core and the
bundle must be evaluated in ONE eval, not two.

**P2's decisions.** Filters come from the shared registry, per strategy.
**They live in the MAIN COLUMN, not the sidebar** (moved 2026-09-25): the
sidebar is ~400px and permanent, filter editing is occasional and wants
width, so nine-and-growing metrics stacked one per row there were squeezing
the thing that is always on screen. A sidebar row's *Filters* button opens a
panel under the summary table — which is **sticky**, so every number stays in
view while a slider moves; watching the table move IS the filtering. The
panel is `x-if`, not `x-show`: hidden-but-present would evaluate every
expression in it against the null strategy. **qty and capital stay inline in
the strategy cards** — they are adjusted repeatedly while watching the table,
which is the opposite case — but on a **second line** beneath the name, with
labels. Laid out as right-hand columns beside the name they overlapped it:
400px does not hold a name, two number inputs and a button abreast, and the
name is the part that cannot be truncated away. The controls themselves (`ob-dual`, `ob-checks`)
are the OO page's, moved into `backtest.css`; slider bounds come from each
strategy's own values rather than the registry's nominal range.

**The Filters button is NOT gated on the portfolio being loaded.** It was
(`:disabled="!loaded.length"`), and that was a regression a user hit: the
button looked normal and did nothing, because `.bp-btn:disabled` kept the
text colour. Filters set before a load are applied by the load, so there is
nothing to protect; what the panel cannot show yet is RANGES, and it says
which of the two it is ("load the portfolio to see this strategy's values"
vs "no values in this log"). `ensureFilters()` back-fills any metric key a
strategy's filter map lacks, so a strategy added before `/registry` answered
cannot leave a hole that the panel reads into a TypeError.

**Every check of that panel passed while the button was dead**, including a
browser check that read its geometry — because they all opened it by calling
`toggleEdit()`. `scripts/check_portfolio_ui.py` now drives the page **by
clicking**, in headless Edge, and captures `window.onerror` and
`console.error`: an Alpine expression error does not stop a page, it logs and
leaves the control inert, which is this exact bug's shape. It found a second
one immediately — closing the panel logged an error on every click, because
Alpine re-evaluates an `x-for` inside a dying `x-if` with `editingRow()`
already null. Registered `can_skip`: no browser on the VPS means SKIP, not
PASS. A filter
drops trades with no value for it and the row states the cost (staggered
coverage). The TOTAL row pools the
filtered, qty-scaled trades and runs the same `obStats`/`obExtraStats` over
them. Two figures need their own definition at portfolio level and the page
says so under the table: **ann ret %** divides by the peak of the SUMMED
deployed-capital series (lower than the sum of per-strategy peaks unless they
all peak together), and **avg P/L %** pools per-trade percentages, each
against its own strategy's capital — which reduces to the single page's
definition when there is one strategy. **Sharpe is the old app's**, labelled
in the header tooltip and under the table: P/L summed by close date,
mean/stdev × √252, days without a close NOT zero-filled, dollars and no
risk-free rate, so it ranks rows against each other and is not comparable
with a published Sharpe.

**Settled with the user (2026-09-24), not to be re-opened:**
- **Our stat definitions win** over the old app's, unchanged — Max DD,
  Calmar, all of them, exactly as the single-backtest page computes them.
- **P/L is dated by CLOSE**, the same basis as the single page, so one
  strategy reads identically on both. No accrual, no switch, no third basis.
- **qty scales P/L linearly**, applied in the browser. Nothing server-side
  multiplies a P/L — two places that scale is a portfolio silently squared.
- **Capital** seeds from the strategy's saved `capital_per_position`,
  overridable on the page; portfolio capital is the SUM of per-strategy
  planned capital (× qty).
- **Date range** defaults to the UNION of the loaded spans, with
  intersection available.
- **Sharpe** is kept from the old app, with the rolling Sharpe/Sortino
  section (P6).
- **Surface metrics are not in v1.**

**P3's decisions.** Equity and drawdown sit SIDE BY SIDE sharing an x axis
(stacked, the pair is two screens apart and the trough no longer lines up
with the dip). The curves are built in the SAME pass as the table, from the
same filtered indices — a second pass that re-derived them could disagree
with the numbers directly above, which is the one thing a chart beside a
table must not do. `obDailyCurve` reduces the trade-level equity to one point
per close date, keeping each day's day-END cumulative and its WORST
drawdown: the worst trade-level point always falls on some day, so the
chart's trough IS the table's Max DD rather than a shallower day-boundary
reading of it (the gate asserts the two strings match). Only the PORTFOLIO's
drawdown is drawn — one curve per strategy on the same axes is a picture of
nothing in particular. Capital deployed is the summed per-strategy series,
stepped (it changes at a close), and its peak is the figure `ann ret %`
divides by. Monthly P/L is a DOM grid, not a canvas: twelve cells a year is
nothing to lay out and the numbers want to be readable; shade is the month's
size against the biggest month, the same opacity rule the bar charts use.
Everything is dated by CLOSE.

**The months and the annual bars are ONE grid** (2026-09-25). The bars were a
Chart.js canvas beside the table, and two elements placed side by side keep
their own vertical rhythm: a year's bar sat at a different height than that
year's row of months, and any change to either would have moved it again. Now
each year is one grid row carrying its twelve month cells AND its bar, so the
alignment is structural rather than tuned — `yearBar()` returns percentages of
the bar's own cell, zero centred only when some year lost money (reserving
half the column for a direction nothing uses halves every bar's resolution to
draw white space). The per-year TOTAL beside December is gone: the bar is that
figure, and printing both was the same number twice. Month tracks are
`minmax(0, 1fr)` — **never a fixed min-width**, which is what forced the
horizontal scrollbar — so twelve FULL figures fit without abbreviating and the
columns flex instead of overflowing.

**Measuring "does it fit" by asking for a scrollbar does not work**, and cost
two wrong gates before the right one. `.bp-mwrap` overflows visibly, and a
visible overflow reports `scrollWidth === clientWidth`, so the check passed
with the grid hanging 400 px out of the card. Comparing element edges passed
too: a grid item with a min-width overflows its TRACK while the container
keeps its width, so the grid's own right edge never moves. What the fault
actually produces is cells crossing each other and the bar column — so the
assertion is **overlap between neighbours, measured at three widths**
(`check_portfolio_ui`), and it is the one that fails when the min-width comes
back.

**A filter that cannot see the whole history no longer shortens the charts**
(2026-09-25). A filter either JUDGES a trade or is BLIND to it, and the two
are not the same thing. Day of Week judges every trade ever, so a Tuesday
excluded by a Monday filter is genuinely gone. VIX can only judge from
`index_ohlc`'s start and a surface z-score from its own; before that there is
nothing to judge by. `obApplyFilters` drops nulls, so a VIX filter silently
cut the equity curve back to 2017 while the date-range card went on
advertising 2013 — which is what made a filtered portfolio look like a
truncated one. Now **equity and drawdown draw the whole span** and HATCH the
stretch the filter was blind to.

- `obApplyFilters(cols, n, specs, lenient)` takes an optional set of columns
  whose nulls mean "no data to judge by" and therefore PASS. Omit it and the
  behaviour is exactly what it always was; the summary table always omits it.
- Which columns: active filters with a registry `minDate`. A metric with
  gaps but no coverage start (a missing `premium`) is NOT blind — those
  trades are dropped from the charts too, and a gate plants that.
- The shade ends at the **last unfiltered trade to CLOSE**, not at the
  coverage date, because the curves are drawn on a close-date axis: a trade
  entered before coverage can close after it, and ending at the coverage date
  would leave it drawn unfiltered OUTSIDE the shade. Gated as an invariant —
  every trade the chart adds back closes inside the shade.
- **A metric column holds TWO kinds of null and only one of them is a
  stretch** (2026-09-25). Everything before the metric's coverage begins, and
  a scattered few after it where the entry had no bar (a 09:30 entry). So
  `lenient` is a Map of column → COVERAGE DATE, and a null passes only for a
  trade entered before it; a no-bar trade is dropped from the charts exactly
  as the table drops it. Treating both as blind let a single late no-bar
  trade drag the shaded stretch to the last close in the series — six years
  past the coverage date, reported as "the shade ends 2026-09-11" on a log
  whose coverage starts 2018-06-08. The gate now asserts the shade stops
  within one holding period of the coverage date and nowhere near the end of
  the data, and the fixture carries both kinds of null. The key states the
  no-bar count separately, so the trades that are neither shaded nor counted
  are accounted for rather than missing.
- With several blind filters active the reason date is the **latest**
  coverage among them, not the earliest: nothing before it has passed all of
  them.
- **The charts and the summary now disagree on purpose.** While anything is
  hatched the curve ends above the table's Total P/L and the trough may be
  deeper than its Max DD. This is the documented exception to "built in the
  same pass from the same filtered indices" below; the card says so, and the
  gate asserts they match only when nothing is hatched. The summary keeps the
  strict trades, per the brief — there is no toggle.
- **All three charts in the pane draw the whole span** (2026-09-25).
  Capital deployed was clipped to the table's trades on the argument that it
  answers what would have been at risk UNDER the filter; three charts side by
  side behaving differently is worse than that inconsistency, so it is shaded
  like the other two. Its TABLE figure stays strict, so the drawn peak can
  run above **peak deployed** — and **ann ret %** divides by the table's
  figure, not the chart's. The card says so.
- **Two shades, THREE tones.** They compound where both hold, so the key has
  a third entry for the overlap whose colour is `1 - (1-a)(1-b)`, computed
  rather than a third constant to keep in step. Without it the darkest region
  on screen matched nothing in the key.
- **Both shadings are the same grey at two densities, and each is a BINARY
  state** (simplified 2026-09-25). Strategy coverage: either every loaded
  strategy is live or some is not — no band for "1 of 3" against "2 of 3",
  and bands merge by that flag rather than by count. Metric coverage: either
  every active filter could be evaluated or one could not — no per-metric
  treatment. Grading them invited the reading that the shade MEASURED
  something; an amber hatch (the first attempt) read as a different KIND of
  information, because blue and pink already carry profit and loss on every
  chart here. Where the two overlap they simply compound and the stretch is
  darker; that is not special-cased, because "neither condition holds here"
  is what darker should mean. `BP_SHADE` is the one colour, with
  `BP_SHADE_STRATEGY` and `BP_SHADE_METRIC` its two densities.
- Each has its own one-line key, and the metric key names the filter that
  SET the boundary — the latest coverage among the active ones — rather than
  listing every blind filter, because the boundary is one date and one
  metric put it there.
- **Ask before adding a colour or a pattern the page does not already use.**

**The date-range dropdown still knows nothing about metric coverage** — it is
`bpSpan()` over each strategy's `date_min`/`date_max` and nothing else. After
this change the tagline matches what is drawn: union draws the union, and
intersection genuinely clips because a date is a thing every trade has.

**P6's decisions.** The **distribution** is the old app's overlaid
histograms — one series per strategy at 0.6 opacity, **$100 bins** as it had
them, on the filtered qty-scaled trades; one bin set is computed across every
strategy so the bars line up (Chart.js `grouped: false` overlays rather than
interleaves). **Strategies active per day** draws a line per strategy plus a
dotted portfolio total, as the old app did — but on OUR concurrency
(half-open, per SPX session, no dedupe by open day) rather than the old
inclusive, deduplicated calendar count: two charts on one page disagreeing
about what a position is would be the same fault as the old app's weekly
matrix beside its daily rolling correlation. **Rolling risk** keeps the old
definitions exactly (mean ÷ stdev × √252, ÷ downside stdev for Sortino, win
rate as the share of those days that made money) with the 30/90/180 selector
and win rate on a second axis, because the old app had two axes and a ratio
does not share a scale with a percentage. **The window counts days that had a
close**, not calendar days — at a couple of closes a week, 90 of them is
closer to nine months than three, and the card says so. A window longer than
the portfolio's close-days draws nothing and says which it is: without that
the empty chart's linear x axis falls back to zero and renders as "1970",
which reads as broken rather than as a window that does not fit.

**The sidebar card carries its ACTIVE FILTER BADGES** — "VIX Level: 12.0–30.0",
"Day of Week: Fri" — ported from the old app's `METRIC_BADGE_COLORS` (pale
ground, darker same-hue text) and keyed by registry key, with the old
wording rules: a range shows one decimal, a categorical with everything
chosen shows just its label (all of them is not a narrowing), and a long list
shows the first two with an ellipsis. Without them you cannot tell which of
several strategies is filtered without opening each panel in turn, which is
the point of having them side by side. **This was in the brief and was
missed, not deferred** (2026-09-25) — as were the annual bars beside the
monthly grid (since rebuilt into that grid, above), and the old app's
arrangement of the correlation card (metric
and strategy tables side by side, scatter full width beneath). All three are
now built.

**Known departures from the old page, agreed rather than missed:** the
Strategy Builder is gone (qty and capital are inline in the cards, filters
are the main-column panel); the date range is a union/intersection toggle
where the old app had two date pickers; profile saving is an inline name box
with a 409-driven Replace prompt where the old app had a modal with a
checkbox; the summary table has 16 columns against the old 13 (our
definitions won); the rolling pairwise window is in WEEKS. **Still to build:
P6** — P&L distribution, trade overlap, and rolling risk metrics (the old
app's 30/90/180 selector, to be decided in weeks or days).

**Surface metrics as per-strategy filters** (2026-09-25/26). Any of the 452
`surface_metrics_core` columns can be added to ONE strategy's filter panel;
it appears on no other strategy's. Values come from the OO page's own
`/surface/values` for every trade of that strategy (so later filter changes
cost nothing) and are cached per `(strategy, metric)`; the catalog comes from
`/surface/catalog`, lazily on first panel open, since it is a ~4s index walk
on the VPS. `registryFor(c)` is the shared registry plus that strategy's
added metrics and replaces `this.registry` at every site that is about ONE
strategy. The pooled metric/PL correlation keeps built-ins only: pooling a
metric one strategy has would report what the portfolio moved with from a
third of its trades. **The coverage cost is stated before the slider moves**,
split into "entered before its data starts" and "no bar at the entry time",
because a surface filter silently dropping a third of a strategy is the
hazard the whole feature carries. A profile stores the metric LIST, not the
values (`clean_surface`, shape-checked only — the catalog owns what exists),
and re-fetches on load; without the list the saved filter came back as a
value describing nothing, since `bpSpecs` walks the registry and not the
filter map.

Three bugs the browser gate caught here, all of which would have shipped:
`scaledCols` caches a COPY of the columns so a column added after it was
built was invisible to every filter — and a range spec drops nulls, so the
filter did not narrow the strategy, it emptied it; `obClampStep` rounds
through `toPrecision(12)` so a ceil can land a float's breadth BELOW the true
maximum, and a slider at its own maximum dropped the trades that set it
(extents now snap outward and never inside, which fixes the built-in sliders
too); and Alpine re-evaluates an `x-if`'s children as it tears down, so every
`key().field` read needs the `(key() || {})` guard `rangeOf` already had.

**P5: profiles are POINTERS, not snapshots.** A profile
(`backtest_portfolio_profiles`, JSONB) holds the strategy ids with their qty,
capital and filters, plus the range mode and rolling window — no trades and
no parse, so the saved strategy stays the one source of its file and
re-saving it there is picked up on the next profile load. The cost is stated
rather than discovered: `GET /profiles/{id}` reports which ids no longer
exist, and the page says "2 strategies have been deleted since" instead of
quietly loading a smaller portfolio. A name collision is a **409 carrying the
existing id**, so the page offers to replace THAT profile rather than asking
for a name it already knows is taken. The payload is validated into shape on
the way in (ids, positive qty, non-negative capital, registry-shaped
filters); **filter keys are stored as given** — the registry owns which
metrics exist and a second list here would drift from it.

**Two bugs P5's browser check caught, both invisible to source reading.**
`loadProfile()` held `busy` while calling `load()`, and `load()` refuses to
run while busy (its double-click guard) — so a profile restored every setting
and then fetched nothing. And `rangeOf(m)` returned `null` for a torn-down
panel, which Alpine evaluates once more on the way out: the expression error
does not stop the page, it leaves the rest of that render pass stale, which
is how a filter panel took the summary table down with it. `rangeOf` now
always returns a shape with a `missing` flag.

**P4's decisions.** **Both correlations are weekly** — the old app's matrix
was weekly and its rolling pairwise was daily, so the two disagreed about
what a correlation is. Strategies close on their own schedules, so a daily
series is mostly zeros and correlating mostly zeros measures how often two
strategies happened to close on the same day. The week is the SUNDAY ending
it, matching pandas' `'W'`, so the buckets are the old app's. Weeks where
every strategy is flat are dropped: matching zeros would pull every pair
toward +1. The matrix is Pearson; **Metric vs P/L is Spearman** (rank, ties
averaged, blank under ten values) because a monotone but non-linear relation
is what a metric usually has. That table pools every strategy's filtered
trades, which mixes strategies that traded at different times — it says what
the PORTFOLIO's P/L moved with, not what any one strategy's did, and the note
under it says so. The rolling chart draws every pair over a window of WEEKS
(13/26/52) with the selected pair in white.

**The filter panel is the TOP of the main column**, above the summary — the
old app's arrangement: configure, then read the results below. I first put it
underneath and justified it by wanting the table visible while filtering;
making the table sticky had already solved that, so the justification did not
hold. **When a brief says to match a layout, match it** and raise conflicts
before building.

**Numbers are never abbreviated.** `$31,247`, not `$31k` — in the table, the
monthly grid, the year totals, the axis ticks and the tooltips. Two rows that
abbreviate at different thresholds cannot be compared at a glance, which is
the whole job of a summary table; if something stops fitting, widen it. The
UI gate asserts no `$…k` or `$…M` appears anywhere on the page.

**Switching a categorical filter on keeps everything.** `bpSpecs` skips a set
filter with no members (matching the old app), so "on with nothing chosen"
was an ACTIVE-looking filter that changed nothing — the inert-filter failure
the scalp page already has a line about. Enabling one now selects every
category, and you untick what you do not want; if you untick them all the
cell says "nothing chosen — not filtering" rather than implying it keeps
none.

**The parse cache** (`app/oo_backtest/parsed_cache.py`). A 2,100-trade
MesoSim log is ~57 MB and takes **3.4 s** to parse here; five strategies is
17 s cold, and the VPS is about half this speed. So the PARSED FRAME is
cached beside the file: encode 0.7 s, decode **0.04 s**, blob 379 KB (the
gzipped source file is 4.5 MB), so a warm five-strategy load is ~0.2 s of
parse work. **Keyed on `file_sha256` + a fingerprint of `data_loader.py`'s
own source** — derived, not a constant to bump, because the failure it
prevents is a stale parse that looks plausible. **Only the parse is cached**;
the market join runs every time, since `index_ohlc` is backfilled and a
frozen join would pin a trade's VIX to whatever the table held on first load.
`df.attrs` travels with the columns — the parser's notes (open positions, the
BacktestName, the data-quality flags) live there, and a cache that carried
only columns dropped them silently. The gate compares **payloads**, not
frames, because the payload is what reaches the browser.

**One load path.** `/api/backtest-portfolio/load` enters
`oo_backtest._analyze` with the cached frame; the OO page's own saved-load
uses the same cache. A second parse/join/payload path would be two sets of
trade numbers for one file. **One stylesheet**: `static/css/backtest.css`
holds the shell (layout, card, type scale, controls, stat grid) both pages
wear — moved out of `oo_backtest.html` inline CSS, verified by rendering the
OO page before and after to an identical PNG.

**Where the old app differs from us** (for P2; ours wins unless noted):
- `avg_annual_pct` — old: `total_pnl / (planned_capital × qty) / years`,
  ignoring concurrency. Ours divides by **peak deployed capital** (peak
  concurrent × capital). The old figure reads ~5–8× higher at 5–8 concurrent
  positions.
- `years` — **close-to-close on both now** (changed 2026-09-25). Ours was
  open-to-close, which was never a decision: it stretched the window by the
  first position's holding period, inflating `years` and so understating
  **avg annual P/L**, **Calmar** and **ann ret %** — all three scale by
  exactly `years_old / years_new`. Close-to-close matches Max DD, Sharpe, the
  axis the equity curve is drawn on, and what the old Render app did. The
  parity gate's pandas reference moved with it, and a filtered set whose
  trades all close on one day now reports "—" rather than annualising a
  zero-length window.
- `calmar` — old is `avg_annual_pct / (|max_dd| / capital × 100)`, which is
  algebraically **the same number as ours** (capital cancels). No conflict.
- `sharpe` — `mean/std × √252` over days that HAD A CLOSE, not zero-filled
  calendar days; at ~2 closes a week that annualisation is generous. Kept as
  the old app had it, per the user.
- Old has no Profit Factor, Avg Days, Max DD %, Avg P/L %; we have all four.
- `capital_deployed` — **settled 2026-09-25: half-open `[open, close)` on
  BOTH pages.** The measure is OVERNIGHT capital, what is still at risk at
  the close, so a position is deployed on the sessions it is held through and
  not on the one it closes on. A strategy entering every Friday and closing
  the next Friday reads a constant 1, where inclusive counting drew 2 every
  Friday — the same position counted twice on the day it changes hands. The
  OO page's Deployment pane was wrong and now matches; its **peak concurrency
  falls**, so **Avg Annual Return % on that page changes too** (it divides by
  peak deployed capital). A trade opened and closed in one session is held
  overnight never and contributes nothing: correct for this measure, and a
  0DTE log therefore draws a flat zero, so `obConcurrency` counts those trades
  (`sameSession`/`counted`) and the pane says so instead of drawing nothing.
  **The old app's dedupe-by-open-day is NOT carried over**: it grouped every
  trade opened on one day into a single slot lasting until the latest of their
  closes, which assumes same-day rows are one position split across rows. In
  our data one MesoSim row is one PositionId, so genuine same-day entries
  would be undercounted, and a short trade opened beside a long one would be
  counted as deployed after it had closed.
- Strategy correlation — old uses **weekly** resampled P/L (deliberately, to
  avoid the daily zero-fill artifact) but its rolling pairwise correlation
  uses DAILY rows, so the two disagree about what a correlation is. **Both
  are weekly in P4** (agreed 2026-09-25).
- `SPEARMAN_METRICS` is a hardcoded list including the dropped SharpTwo/skew
  metrics; we use the registry.

## The topbar nav: six categories (2026-09-24)

One partial, `templates/_nav.html`, included by all 17 page templates — it is
the only file with nav markup, so this was one edit rather than seventeen.
Seventeen links in a row became six disclosure menus: **SPX · Factor · Equity
· Scalp · Backtest · Research**.

- **A category name is a button, not a link.** It has no page of its own, and
  the alternatives are inventing a landing page or promoting one child.
- **`nav_active` is unchanged and is the KEY, not the href.** Every page
  passes the same string it always did, so no page template and no route
  moved. The three Equities pages are on the other service and their key
  (`/equities-live`) is deliberately not their path there (`/`), which is why
  an item carries `live` separately; the href is still built from
  `request.url.hostname` and `live_port` (the box is reached by both Tailscale
  IP and name).
- **Plain JS (`static/js/nav.js`), not Alpine**: every page has its own
  component on `<body>`, and a nested one here would be a second thing for
  every page's gates to know about. The W3C *disclosure navigation* pattern —
  `aria-expanded` on the button, ordinary links Tab reaches — not
  `role="menu"`, which takes links out of the tab order and is for
  application commands.
- **The bubbling guard is load-bearing.** The category listener sits on an
  element containing the button, so a key the button handled arrives again
  with focus already in the menu: Down opened the menu, landed on the first
  page and stepped straight to the second, and **every arrow moved two**.
  `if (e.target === b) return;`. Only a browser could see it — found by
  driving the keyboard in headless Edge, and the gate now checks the guard is
  still there.
- The current page's category wears the accent; the page is marked again
  inside its menu, with a `::after` dot so it is not marked by colour alone.

Gate: `scripts/check_nav.py` renders every page and asks a person's
questions — is this page's key in exactly one category, is that category the
marked one, is the page marked once inside it, are the names buttons, do the
hrefs go anywhere, and does the nav link to any key no page claims (which is
what a renamed route leaves behind).

## Equities Scalp: the filter pane (2026-09-23/24)

**Two defects in a row, the first concealing the second.** Worth reading
together, because the second was invisible until the first was fixed.

**1. The ranges (2026-09-23).** `col_ranges` is keyed two ways, deliberately:
the pivot's columns under their ROLE key (`noise`, `ratio`, `price`), with
ranges taken from the rows the table is drawn from; and every metric the date
holds under its OWN NAME, from one grouped aggregate over the date
(`min`/`max`/`percentile_cont(0.5)`/`count`). The pane lists metrics by name
(from `/meta`, queried live) and looks each one up by name. Before this, only
the role-keyed columns had ranges, so every metric row read **"not on this
date"** with its ≥/≤ buttons disabled, and the only working row was the
DERIVED one (`$ vol/min`), whose key is its own. Live since `c6f4741`, which
introduced the pane — nothing to do with the metric cull, though the symptom
invites that reading. A metric that is present but entirely null still gets
no range and says so in the same words: there is nothing to screen on either
way.

**2. The constraints were never sent (2026-09-24).** `loadCandidates()` built
the query without `filters`, so a pane constraint lived only in the browser:
the chip appeared, the number was accepted, and the count never moved —
`> 0.40` and `> 40` behaved identically because neither was asked about. The
endpoint's half was complete all along (parses `filters`, pulls the named
metric into the pivot, evaluates it in the same loop as the sliders, reports
it inert if it could not run). It was hidden by defect 1: with every ≥/≤
button disabled, no pane constraint could be created — and `$ vol/min`, the
one metric that escaped, has a named SLIDER, and sliders were always sent.

**Gates, one per half.** `scalp_dryrun.check_filter_pane_ranges` asks the
page's own question — for each metric `/meta` lists, is there a range under
that exact key — and then that screening on an unchosen metric runs, is
reported active rather than inert, and rejects.
`scripts/check_scalp_filters.py` drives the SHIPPED page JS in node with the
network stubbed and reads the URL it builds; a source check would have passed
against the broken version, since the word "filters" appears throughout the
file.

## Equities Wall (`/wall`, in progress — 2026-09-22)

A page of dozens to ~100 small live tapes, so a whole watchlist can be
scanned at a glance for which names are quiet and which have a usable
spread. **It is for watching.** No ladder, no arming, no order entry,
nothing that reaches a broker — gated (`check_wall.py` parses the wall's
modules and its endpoints and refuses any reference to the broker).

**No metric, no score, no colour-coding, no ranking.** The scan page's metric
surfaced untradeable names; this page shows the picture and lets the eye
decide.

**Phases:** P1 server side (hub tier, watchlist store, endpoints, frames) —
done; P2 the page and its canvas panes — done; P3 controls — done; P4
capacity (the user's own measurement on the box) and docs.

**The vertical scale is the point.** Each pane scales so the spread fills a
fixed share of its height (default 60%, centred on the mid, adjustable, with
a per-pane override saved next to the ticker). "Bouncy" is defined relative
to the spread's own width, so a half-spread move looks the same on FDX (7c)
and LLY (50c), and absolute spread width stops dominating the wall. The
spread number in cents is what tells them apart.

**Decisions already made and gated:**
- **A third tier on the one upstream socket.** The account permits exactly
  one Polygon connection; a second evicts the first. `Hub.holders(sym,
  exclude=tier)` is the single place that answers "does anyone else still
  hold this", and every tier's add and drop goes through it — the failure it
  prevents (an unsubscribe pulled out from under another tier) shows as a row
  or pane that has gone QUIET, not blank, which is the one distinction these
  pages exist to make.
- **One browser connection for the whole page.** The service caps browsers at
  `MAX_CLIENTS` (8); a hundred panes with a socket each is twelve times over.
  `/wall/ws` carries every pane.
- **One frame a second, and a symbol with nothing new is not sent** (which is
  what lets the page skip that pane's redraw). The frame still goes out when
  empty, or the page cannot tell a quiet market from a stopped feed.
- **The cursor is a COUNT, not a timestamp.** Several trades share a
  millisecond; a timestamp cursor drops all but the first, silently, on
  exactly the names being watched.
- **The quote is sampled at 200 ms for the band, but `last_quote` takes every
  message** — the spread number is a fact about now. A crossed or one-sided
  quote is dropped and counted (a zero bid would throw the pane's whole
  scale). The typical spread is a **median over the last minute**, so one
  wide quote cannot rescale a pane; with no samples in the window it falls
  back to the last quote, because a quiet name's spread is old, not absent.
- **The watchlist lives server-side** (`data/wall_watchlist.json`, written
  whole through a temp file and a rename) and carries each entry's scale
  override, because the override is a fact about that symbol on this wall.
  `WallRunner.apply()` is the one path that sets the hub tier, the file and
  every open page — two of the three would be wrong in a way nobody sees
  until the next deploy.

**The page (P2/P3).** Canvas per pane, unlike the scan page's windowed DOM:
what is drawn is a picture (a couple of hundred bubbles and a stepped band)
rather than a row of cells, and there is nothing to hit-test. One
`setInterval` at 1 Hz draws every pane; a pane is skipped when it is off
screen (IntersectionObserver) or has nothing new, with a forced refresh every
`WL_FORCE_REDRAW_MS` (5 s) so a quiet pane's bubbles cannot sit still while
the window slides out from under them. Prices are drawn against the SERVER's
clock (`at` minus local elapsed), not the browser's; a laptop a few seconds
out would put every print off the pane.

**A wall pane is a small Equities Live pane, and the colours are shared.**
`static/js/tape_theme.js` holds them — blue bid, pink ask, neutral
translucent prints with a rim on discs over 2px — and BOTH bundles read it
(`equities_live.js` keeps its `LV_*` names but assigns them from `TAPE_*`).
A colour written out twice is two pages that agree until one is edited, and
the drift is invisible until they are open side by side, which is how this
page is used. The shared file loads in its own non-deferred `<script>` BEFORE
each bundle: the constants are top-level, so the wrong order is a
ReferenceError and a blank trading page, and the gate checks the order,
executes both bundles beside the file, and asserts the tape bundle FAILS
without it. `check_live_axis` and `check_live_reconcile` prepend the same
file in their node drivers. **No fill between the bid and the ask** — two
lines and nothing between, as on the tape page; a shaded band made the wall
read as something else at a glance across a hundred panes.

**`x-init="init()"` IS A DOUBLE INITIALISATION.** Alpine 3 calls a data
object's own `init()` automatically, so naming it in `x-init` as well runs it
twice — on this page that was two WebSockets and two draw loops per tab,
found by rendering the page in headless Edge and counting the sockets (2,
then 1 after removing it), invisible to every source-reading gate. **Every
other page in this app still does it** (14 templates, `ai_explorer`,
`equities_live`, `equities_scan`, `oo_backtest`, … all pair `x-init="init()"`
with a component that defines `init()`): each one double-fetches on load, and
the tape page holds two of `MAX_CLIENTS`' eight slots per tab. Not changed
here — it is a separate, wider fix.

**The spread floor FADES, it does not filter** (`min_spread_cents`, saved
with the list, 0 = off). A pane under the threshold is drawn at 0.30 opacity
IN PLACE: every name keeps streaming and keeps accumulating, so a symbol that
dips under and comes back has its two minutes intact instead of rebuilding,
and nothing downstream of the control reaches the hub (gated: setting it adds
and drops nothing and sends nothing upstream). Panes never reflow — the grid
iterates the whole list and marks the failures, because on a wall you
recognise positions, not names. Two rules stop it flickering: it compares the
**typical** spread (the server's median of the last minute, already sent as
`tp`), never the instantaneous `sp`; and it **dims late, undims at once** —
ten seconds below before fading, full strength the moment it qualifies. A
threshold CHANGE back-dates the timer so the slider applies immediately (a
control that does nothing for ten seconds looks broken). A symbol that has
never quoted cannot be shown to qualify, so it fades with the rest and its
header reads "—". The status line names the count ("9 under 10c") because a
faded pane is set aside, not dropped.

Controls: add box and an Edit-list textarea (an edit KEEPS each symbol's
override), window, spread-share and spread-floor sliders, pane width. The per-pane override
is not a control on every pane — a hundred panes carrying sliders is a
hundred controls on a page whose job is to be looked at — but a click selects
a pane and the bar grows a scale slider, "use default" and "remove" for it.
Window and share are saved server-side with the list (debounced 500 ms); pane
WIDTH is local storage, because it is about this screen and the same list is
read on a laptop and a 32-inch monitor.

Files: `live/wall.py` (the per-symbol store), `live/wall_store.py`,
`live/wall_runner.py`, the wall tier in `live/hub.py`, `/wall/*` in
`live/main.py`, `templates/equities_wall.html`, `static/js/equities_wall.js`.
Gate: `scripts/check_wall.py`, 29 cases, no market needed — the last five
execute the shipped JS in node (the scale property: a half-spread move is the
same fraction of the pane on a 7c name and a 50c one).

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
  Not data. The calendar names them for what they are: values written while the market was
  shut, reported as artifacts with no threshold involved.
- **Postgres NaN:** NaN sorts above every number. `x <= 0` is false for NaN, `x > 0` is
  true, `x = 'NaN'` is true. A validity test needs `> 0 AND <> 'NaN'` (not null, not NaN,
  positive). Make values valid *before* any aggregate — `max()` over a column with one NaN
  returns NaN. This bit three separate times.
- **Coverage starts:** SPX 2017-01-03, VIX 2017-01-03, VIX3M 2017-10-24, VIX9D 2018-06-08.
- **2026-04-08** is a real trading day (full VIX session) with zero SPX bars — missing
  data, not a non-session.

### The exchange calendar decides what a session is (2026-09-25)

**This reverses "no trading-day calendar anywhere", which was wrong.** The old rule —
`MIN_SESSION_BARS = 34`, a series has a session when it has ≥ 34 valid bars in
09:30–15:55 — was reverse-engineered to sit between a real session (78 bars, 42 on an
early close) and the 11–25 stale bars VIX ingestion writes on holidays. The argument for
it was that a calendar would disagree with the table on days like 2026-04-08. **It
disagreed because the table was wrong**: 2026-04-08 was an ordinary NYSE session with a
full VIX session and zero SPX bars — a missing day in the upstream pipeline that the rule
reported as "not an SPX session", indistinguishable from a holiday. Do not restore the
data-derived rule as a cleanup.

**Two questions, answered separately.** `app/oo_backtest/market_calendar.py` (NYSE, via
`pandas_market_calendars`, rules bundled — no network) says whether a day was a SESSION.
The bars say whether a SERIES has usable data on one. `MIN_SESSION_BARS` is gone.

- **Expected bars come from each day's own close**: 78 for 09:30–16:00, 42 for a 13:00
  early close, so a SHORT half-day is detectable where a single floor could never see one.
  `BAR_SHORTFALL_TOLERANCE = 2` is set by this table's own behaviour — VIX3M and VIX9D can
  arrive a bar or two behind.
- **The rollup is one row per exchange session** in the table's range, sessions and
  expected counts passed in as two arrays. A session the writer left empty is a row with
  nulls and its flags off, not an absent day.
- **`session_report` reports three things apart**: non-session days; bars written while the
  market was SHUT (an artifact — no threshold needed to recognise one); and per series on a
  real session, complete / partial / **missing**. Shortfalls are grouped by size ("short by
  53 on 1 session") and a series with no data on a session gets a log WARNING naming the
  dates. It is a pipeline hole, not a quiet day.
- **The deployment axis is a market fact**: `session_days` asks the calendar, so it no
  longer stops where `index_ohlc` starts (a 2013 strategy had four years of blank capital
  deployed beside a full equity curve) and a session with SPX missing is still a day, since
  a position was held over it either way.
- **Freshness is counted in COMPLETED sessions** (`STALE_AFTER_SESSIONS = 1`), not five
  calendar days. A session that has not finished is not counted, so it cannot fire every
  afternoon; weekends and holidays cost nothing, which is what the five days used to buy.

Sessions are still **per series** for DATA: SPX's previous close is from the previous
session SPX had usable data on, VIX's from the previous session VIX did.

`scripts/check_market_calendar.py` (offline, runs on the VPS) pins the cases that decided
this, 2026-04-08 by name. The test helper's hand-written holiday list is gone too — a
second, worse trading calendar living in a gate.

**The dependency is hard, and install order matters**: `pandas_market_calendars>=5.0`, and
on the VPS it goes in BEFORE the pull —
`sudo /spx_analysis_dashboard/.venv/bin/pip install 'pandas_market_calendars>=5.0'`. The
module raises ImportError rather than falling back, deliberately: a silent fallback to the
bar-count rule would restore exactly the hiding this removed.

**Known upstream data problems (not page bugs):**
- 2026 16:00 bars stopped: ~249 valid/year through 2025, 65 of 174 in 2026. Ingestion
  changed mid-year. The page uses 15:55, so unaffected.
- VIX hole May–July 2026: 23 session days with no VIX. Causes the 17 null VIX levels and
  clustered null VIX gaps.

### Decisions, and why

- **Max drawdown is evaluated at DAY END** (2026-09-25), in
  `app/oo_backtest/stats.py` and `obEquity` alike, and in the source app so the
  two still agree. Two trades closing on one day at −$5,000 and +$5,000 are not
  a $5,000 drawdown: the other position was open and offsetting, and only the
  day's net was ever at risk. `obEquity` sums P/L per close date and walks days,
  so its points ARE the daily curve — which is why `obByClose` and
  `obDailyCurve` no longer exist, and why the source app's unstable
  `sort_values("date_closed")` stopped mattering: a day's net does not depend on
  the order its trades are summed in. The table's Max DD and the chart's trough
  still agree, both on day end. Magnitude: none at all on a log with one close a
  day, ~1.7% shallower where closes occasionally coincide, ~7% with three a day
  — and most on the portfolio TOTAL, which pools every strategy's closes.
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
  indicator: stale when more than `STALE_AFTER_SESSIONS = 1` COMPLETED
  exchange sessions have passed with no new SPX bar (changed 2026-09-25).
  It was five CALENDAR days, with a note saying a trading calendar was
  deliberately avoided — that is reversed; see the market-calendar section.
  Five days was the price of not knowing which days were sessions, and it let
  a stalled writer go unnoticed across a long weekend. A session that has not
  finished is not counted, so the warning cannot fire every afternoon, and
  the missed sessions are named rather than only counted.
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

**`check_vendored` is green** (2026-09-26). It had failed for weeks on
`scalp_config.py` / `scalp_metric_docs.py`, dismissed as pre-existing. It was
not cosmetic: the copies predated the pipeline's move of its noise statistic
from **rms to p75**, so the page pinned the abandoned column, did not know
about three intraday columns, and showed no definition at all for ten
quiet-window metric families it reported as undocumented. `DEFAULT_FILTERS`
was identical throughout, so the sliders were never wrong — which is probably
why it went unnoticed.

Re-vendoring needed one structural change: the pipeline's `config.py` now
does `from scalp.quiet import ...`, and a VERBATIM copy cannot edit that line
while the rule stands that `rm -rf scalp/` leaves this app running.
`app/__init__.py` binds `scalp.quiet` to `app/scalp_quiet.py`, which
check_vendored already holds byte-identical to the pipeline's copy, so the
name resolves to the same source without reaching outside this repo. It
defers to a real `scalp` package when one is importable rather than
shadowing it.

`_STAT_PREFERENCE` in `equities_scalp.py` now leads with `p75`, not `rms`.
It is the fallback for a date missing the pinned column, and leaving rms
first meant a date carrying the pin showed p75 while a date without it showed
the statistic the pipeline had just abandoned. This matters most on
`intraday_monthly`, which KEEPS its old rms column and stops extending it, so
both statistics live in that table and the order here decides which is shown.

`scalp_dryrun`'s fixture hand-wrote the pinned family as rms and went stale
the same way; it builds those names from `scalp_config` now.
