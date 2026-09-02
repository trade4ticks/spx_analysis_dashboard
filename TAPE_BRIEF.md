# Equities Live — a scrolling tape plot

## What this is

A new navigation page, **Equities Live**, in the nav bar alongside Equities Scalp. Same dark
theme, same accents — blue `#3498db`, pink `#e84393`.

On it: a live scatter of trades. **x is time, y is price, bubble size is share count**, drawn
in bright blue. Behind them, the NBBO as two thin lines. A rolling window of the last few
minutes, scrolling right to left.

That is the dot layer of Bookmap without the heatmap. I want to see the *shape* of the tape —
whether prints are spread across the spread or piled at one price, whether there are gaps
where nothing trades, whether a burst of activity comes with price moving or without.

Separate from the Equities Scalp page. That one ranks tickers overnight from stored metrics.
This is a live view of one name while I am trading it. Different page, different data source,
no shared database.

## Why existing tools don't do this — already investigated, don't redo it

**thinkorswim Time & Sales** shows every print but as a scrolling text list. I cannot see
distribution or clustering in a column of numbers.

**Bookmap inside thinkorswim** is ruled out, and I tested rather than assumed. On FDX, T&S
showed ~60 prints in a minute and Bookmap drew 4–5 dots. I set *Minimal displayed volume* to
1 and *Minimal trade size* to 1 in its Volume Dots configuration and turned clustering off.
The dot count did not change.

The cause is coverage, not settings. Its window title reads `FDX@DXFEED#5` — dxFeed, whose
equity depth is Nasdaq TotalView plus Cboe. FDX is NYSE-listed and its prints land on
exchanges that feed does not carry. Its ladder also showed an identical 220 stacked across
fifteen consecutive price levels, which is thinkorswim's documented "Extended Market Depth"
behaviour: last-known sizes for levels it does not actually have.

No configuration fixes a feed that lacks the trades.

## Data source — verified, use this

**Polygon (branded Massive), Stocks WebSocket.** I confirmed trade coverage myself against a
live FDX session:

```
389 trades over 420s                  → 56 trades/min, matching thinkorswim T&S
under 40 shares: 354/389              → 91% odd lots, sizes down to 1 share
tapes: {1: 389}                       → NYSE tape, correct for a NYSE-listed name
exchanges: 4,10,12,15,17,19,21        → seven venues in the first half-second
```

Every print, every venue, all sizes. The odd lots are the part of the tape I actually
participate in, and every other tool drops them.

**Two channels: `T.<symbol>` for trades and `Q.<symbol>` for quotes.** I am upgrading to
Stocks Advanced ($199/mo), which includes both plus real-time. Build against real-time.

Put the host in config, not inline — `wss://socket.massive.com/stocks` for real-time,
`wss://delayed.massive.com/stocks` for the delayed socket. If the connection is ever delayed,
say so prominently on the page. A 15-minute-old tape that renders identically to a live one
is dangerous.

API key is in `/Open_Interest/.env` as `POLYGON_API_KEY`.

**Trade fields:** `sym`, `p` price, `s` size, `t` SIP timestamp ms, `x` exchange id, `z` tape
(1 NYSE, 2 AMEX, 3 Nasdaq), `c` conditions, `trfi` TRF id.

**Design for the quote volume from the start.** FDX runs ~283 quote records per minute
against ~56 trades — roughly 5x. Retrofitting that into a plot built for the lighter stream
means reworking the rolling window and the render loop, so account for it now even though the
lines can default to off.

**Do not deduplicate or collapse same-timestamp trades.** I saw seven at `.401` and a dozen
at `.402` — single marketable orders sweeping several venues. That clustering is information
about how flow arrives, and pixel-slice aggregation is exactly what made Bookmap useless
here. Plot every trade at its actual timestamp. No time bucketing anywhere.

## Build one pane first

One symbol, one plot, working well. Then an **Add pane** button for a second and third. Do not
build the multi-pane layout first.

## The plot

- **x** — time, rolling window, default 3 minutes
- **y** — price
- **bubbles** — bright blue `#3498db`, area proportional to share count. **Area, not radius**,
  or the small end vanishes. A 1-share print must still be visible next to a 200-share one.
- **no colour by side.** Colouring by bid/ask assumes a two-sided reading I don't use — I buy
  near the bid and sell below the mid almost every time, so both fills sit in the same half.
  Later, once quotes are in, a gradient by distance from the mid might mean something. Not now.
- **NBBO as two thin lines** behind the bubbles, muted so they don't compete. Toggleable.

New trades enter at the right, the window scrolls, old ones drop off the left.

## Controls

- **Zoom x** — widen or narrow the time window
- **Zoom y** — expand or contract the price range
- **Recenter** — a button, and automatic *only when price approaches the edge* of the visible
  range. **Do not recenter on a timer.** A plot that jumps every 30 seconds destroys the visual
  anchor I'm reading position against.
- **Pause** — freeze the view to study something, keep buffering, resume and catch up
- **Symbol input** — type a ticker, the pane resubscribes. No rebuild.
- **Hover readout** — timestamp, price, size, exchange for a bubble
- **Trades-per-minute readout** somewhere on the pane. I use arrival rate as a proxy for
  fillability, and seeing it live tells me when a name has gone quiet.
- **Y axis labelled in cents from the current price**, or at least a visible cent scale
  alongside absolute price. When I'm judging whether a 10-cent capture is available, distance
  reads better than absolute levels.

## Where it runs

VPS at 100.76.94.99, viewed in a browser. One process holding the WebSocket, pushing to the
page.

Memory should be trivial, but **cap it explicitly** — hard limits on panes and window length
so it cannot grow without bound. That box has been OOM-killed twice this week and already runs
three dashboards, Postgres, the ThetaData Terminal and batch jobs.

**Separate service and port from the existing dashboards.** If this crashes I don't want it
taking the others down. It must survive the socket dropping: reconnect with backoff, and say
on the page when disconnected rather than showing a frozen plot that looks live.

## Not now, but don't design them out

- **My own fills marked on the plot.** I have Schwab API access and may eventually trade from
  this page. Seeing my entry and exit against the tape is the direct answer to "was anything
  trading at my price while I sat there," and no tool gives me that.
- A second pane on the same symbol at a longer window
- Gradient colouring by distance from mid

## Explicitly not wanted

- No order book, no heatmap, no depth beyond the NBBO lines
- No aggregation or bucketing of any kind
- No indicators, no derived lines, no volume profile
- No historical replay
- No connection to the equities_scalp database

## Test plan

1. One pane, FDX, market hours. Every print visible, sized by share count, NBBO lines behind.
2. Count prints in one minute against my reference: ~56/min on FDX, ~91% under 40 shares. Fewer
   means something is filtering.
3. Zoom both axes; confirm recentering only fires near the edge, never on a timer.
4. Pause, wait 30 seconds, resume — should catch up rather than skip.
5. Leave running an hour; process memory must not grow.
6. Kill the network briefly — page says disconnected, then recovers.
7. Add a second pane on EXPE; both stream independently, neither stalls.
