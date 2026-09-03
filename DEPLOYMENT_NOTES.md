# Deployment notes

Things about how this app is *served* that are not visible from the code,
and that cost real time to rediscover.

---

## How the box actually serves these pages

**There is no nginx and no Caddy.** A `cloudflared` tunnel named
`dashboards` (`4d1f43f6-c604-46ce-b809-9713135254d2`) maps hostnames
straight to local ports. Each app is its own systemd unit:

| Hostname | Port | Unit | Repo |
|---|---|---|---|
| `iv.pinkbluelabs.com` | 8000 | `spx-dashboard.service` | `/spx_analysis_dashboard`, `run.py` |
| `portfolio.pinkbluelabs.com` | 8050 | `portfolio-dashboard.service` | — |
| `vps.pinkbluelabs.com` | 8080 | `vps_dashboard.service` | — |
| `live.pinkbluelabs.com` | 8001 | `spx-live.service` | `/spx_analysis_dashboard`, `run_live.py` |

`spx-live` is a **separate unit from `spx-dashboard` on purpose**: the tape
holds an upstream WebSocket and redraws thirty times a second, and a crash
there must not take the three dashboards with it.

### The tunnel is REMOTELY managed — `/etc/cloudflared/config.yml` is inert

This cost a wrong conclusion once already. The unit runs with
`--config /etc/cloudflared/config.yml`, so that file looks authoritative.
It is not. On every start cloudflared logs

```
INF Updated to new configuration config="{\"ingress\":[...]}" version=2
```

and serves **that**, which comes from the Cloudflare dashboard
(Zero Trust → Networks → Tunnels → `dashboards` → Public Hostnames). The
proof it is not the local file: the pushed config carries two rules that
have never been in it — `portfolio` and `vps` repeated as
`https://…:443`.

**Adding a hostname is therefore two steps, and step 1 alone looks broken:**

1. `cloudflared tunnel route dns dashboards <host>.pinkbluelabs.com`
   Creates the CNAME. Needs `/root/.cloudflared/cert.pem`, which is present.
2. Add the Public Hostname in the dashboard — or `PUT` the ingress through
   the API with a token carrying `Cloudflare Tunnel:Edit`. **There is no
   Cloudflare API token anywhere on the box**, so today this is a dashboard
   action.

With only step 1 done, the hostname resolves, reaches the tunnel, matches no
ingress rule and gets the catch-all `http_status:404` — an empty 404 that
reads like a routing bug rather than a missing rule.

**WebSockets need nothing special.** They pass through the tunnel unchanged,
and the 100-second origin limit below applies to HTTP responses, not to an
open socket.

---

## Schwab Trader API — things that cost time to rediscover

**The app registration already carries trading scope.** Confirmed by a
read, not by placing an order: `GET /accounts/{hash}/orders` returns
**200**, where a market-data-only registration is refused outright. The
same registration backs the portfolio dashboard's transaction fetch.

### `Content-Type` on a bodyless GET returns 400

Schwab answers a GET carrying `Content-Type: application/json` with

```
400  { "errors": [ { "status": 500, "title": "Internal Server Error" } ] }
```

The identical request without the header returns 200. Nothing in the
message names the header, and the wrapped `500` reads like an outage. This
made *every* read fail starting with the account lookup, and was
indistinguishable from an expired token or a missing scope.

Send `Authorization` alone unless there is a body. The portfolio
dashboard's client always did, which is why the same endpoints worked
there.

### Measured latency, from the box

| Call | Median |
|---|---|
| `/accounts/accountNumbers` | 255 ms |
| `/accounts/{hash}?fields=positions` | 368 ms |
| `/accounts/{hash}/orders` (1 day) | 850 ms |
| `/accounts/{hash}/orders` (2 hours) | 870 ms |

Two things follow. **Narrowing the orders window buys nothing** — the cost
is Schwab's, not the payload's — so the fix for a slow state read is to
overlap the two calls, which takes it from ~1220 ms to ~900 ms. And **a
new `AsyncClient` per call costs ~80 ms of TLS handshake** (280 ms vs 200
ms steady state, 409 ms vs 252 ms cold). One client, reused.

~900 ms is therefore the floor on how fresh the working-order list can be.
`live/config.py: STALE_AFTER_S` is set against that.

### The token file is shared with the portfolio dashboard

`/root/Portfolio_Dashboard/schwab_tokens.json`, and Schwab **rotates the
refresh token on every refresh** — so two processes refreshing at the same
moment leaves one holding a dead one, which is a re-authorisation rather
than a retry.

`live/broker.py` narrows the window: it refreshes only inside the last 60
seconds of the access token's 30-minute life, takes an exclusive `flock`,
re-reads the file after acquiring it, and re-reads once more on failure.
**The lock cannot bind the portfolio dashboard**, which does not take one.
The proper fix is a single owner for the refresh.

---

## Cloudflare's 100-second origin limit (HTTP 524)

**Symptom.** A long request "fails in the browser" while the application
logs `200 OK`. The browser receives an HTML page beginning `<!DOCTYPE`
with the title **"A Timeout Occurred"** and status **524**.

**Cause.** The proxied domain is behind Cloudflare. Cloudflare gives the
origin **100 seconds** to respond. Past that it returns its own 524 error
page to the browser and stops waiting — but uvicorn keeps working and
eventually logs a normal `200`, because from the app's side nothing went
wrong. The two logs disagree because they are describing different halves
of the same request.

**This is not configurable on Free or Pro plans.** Enterprise can raise it;
`proxy_read_timeout`-style settings are irrelevant because nginx is not the
component timing out.

**What crosses it today**

| Endpoint | Typical | Verdict |
|---|---|---|
| `/api/factor-trades/run`, `/zone` | 1–2 s | fine |
| `/api/factor-trades/suite` (N=10, 31 queries) | ~10–20 s | fine |
| `/api/factor-trades/grid` (100+ combinations) | **90 s – 2 min** | **crosses** |

**Fix in use: reach the box over Tailscale, not the proxied domain.**
Cloudflare drops out of the path entirely, and with it the limit. This is
the right answer for a single-user internal tool — the proxy is buying
nothing here that Tailscale does not already provide.

**If the proxied domain must be used**, the options are, cheapest first:

1. Keep runs under ~90 s. Limiting; any larger grid hits it again.
2. Stream the response so headers arrive immediately. Cloudflare starts its
   100-second clock on *the origin responding*, so a response that begins
   emitting straight away does not trip it. Costs a streaming format
   (NDJSON) on both sides and gives real progress as a side effect.
3. Make the endpoint asynchronous — return a job ID, poll for the result.
   Removes the ceiling permanently. See the sizing note below.

### Sizing note for the async-job option

`run.py` starts uvicorn with **no `workers` argument**, so this is a
**single process**. That matters: an in-process job registry (a dict of
`job_id -> {state, progress, result}`) is sufficient and needs no Redis or
DB table. If workers are ever added, that assumption breaks silently — a
job started on one worker is invisible to the others — so the registry
would have to move out of process at the same time.

Roughly: three endpoints (start / status / result), an `asyncio.create_task`
runner, TTL eviction so finished results do not accumulate, and a client
poll loop replacing one `fetch`. The elapsed-seconds progress indicator
becomes a real percentage, because variant completions are countable.

---

## Response compression

`app/main.py` adds `GZipMiddleware(minimum_size=1024)`. The grid payload is
one returns-array per combination, which compresses roughly **8:1** — ~9 MB
becomes ~1 MB. Worth keeping independently of any timeout: it is the
difference between a usable transfer and a slow one on any link.

Bodies under 1 KB are skipped, where compression can make the response
larger than the original.

---

## Client-side diagnosis of this class of failure

`static/js/factor_trades.js` → `_postJson()` inspects **status, then
content-type, then parses** — deliberately in that order.

An earlier version called `r.json()` first and reported every parse failure
as an oversized payload. That is how a Cloudflare 524 page spent an
investigation being treated as a size problem. When the response is not
JSON the helper now prints the status, the content type and the first 200
bytes, and says outright that an HTML body means something in front of the
app produced it.

Keep that ordering. The response usually carries the answer; guessing at a
cause while holding it is the expensive mistake.
