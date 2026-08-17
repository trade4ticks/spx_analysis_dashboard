# Deployment notes

Things about how this app is *served* that are not visible from the code,
and that cost real time to rediscover.

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
