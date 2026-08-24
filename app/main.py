import hashlib
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from fastapi.templating import Jinja2Templates

from app.db import init_pool, close_pool

# Raise multipart upload limit from 1MB to 200MB for backtest file uploads
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_part_size = 200 * 1024 * 1024   # 200MB per file part
    MultiPartParser.spool_max_size = 200 * 1024 * 1024   # 200MB spool
except (ImportError, AttributeError):
    pass
from app.routers import meta, skew, term, historical, concavity, skew_slope, term_slope, raw, heatmap, today, ai_explorer, research, research2, oi_signals, oi_analysis, oi_portfolios, backtest_iv, ticker_analysis, ticker_chain, factor_trades, equity_iv, equity_iv_surface

BASE_DIR = Path(__file__).parent.parent  # project root


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="SPX IV Dashboard", lifespan=lifespan)

# Compress responses. The parameter grid returns ~9MB of JSON (one returns
# array per combination) and JSON of that shape compresses roughly 8:1, so
# this is ~1MB on the wire instead. That matters beyond bandwidth: a reverse
# proxy in front of uvicorn measures its read timeout against TRANSFER TIME,
# and an uncompressed multi-megabyte body over a slow link is what pushes
# past it — the app logs 200 while the browser is handed the proxy's HTML
# error page. Shrinking the body attacks that directly, from inside the app,
# without touching the proxy config.
#
# minimum_size skips small responses, where the CPU is not worth it and the
# compressed form can be larger than the original.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# HTML pages render through Jinja so shared chrome (_nav.html) and repeated
# markup (_macros.html) live in ONE place instead of being hand-copied per
# page. Previously these were FileResponse — a `templates/` directory in name
# only, with zero Jinja tags in it.
#
# keep_trailing_newline is load-bearing: Jinja strips the final newline by
# default, which would make every rendered page differ from its source file by
# one byte. With it set, rendering is byte-identical to the raw file for a
# template that uses no Jinja tags — which is how the conversion was verified
# (see scripts/check_template_render.py; run it after ANY template edit).
#
# StaticFiles above is a separate ASGI mount and is unaffected: /static/* JS
# and CSS keep their ETag / Last-Modified caching. Only the HTML loses ETag,
# which makes the content-hash cache-buster on JS strictly more reliable.
# Starlette >= 1.0 removed the legacy TemplateResponse(name, {"request": ...})
# signature; request comes FIRST now. The old form raises
# "TypeError: unhashable type: dict" at request time, not import time, so
# it cannot be caught by a template-rendering check — see
# scripts/check_routes_smoke.py, which exercises the real route layer.
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
templates.env.keep_trailing_newline = True


# ── Cache-busting off a CONTENT HASH ──────────────────────────────────────
# Templates used to hard-code "?v=NN" and a human had to remember to bump it
# on every JS edit. Forgetting once costs a debugging round: the browser
# keeps serving the previous bundle, the symptom looks like a code bug, and
# nothing about the page tells you which version you are actually running.
# That happened, so the number is now derived rather than maintained.
#
# Hash is of file CONTENT, not mtime: a redeploy that rewrites files without
# changing them must not invalidate a warm cache, and an edit that happens to
# preserve mtime must not be missed.
#
# Cached in-process, keyed on (mtime_ns, size), so the common case is a stat
# rather than a read. A dev editing a file gets a new hash on the next request
# without a restart.
_ASSET_CACHE: dict = {}


def asset(path: str) -> str:
    """URL for a file under /static, with a content-hash cache-buster.

    Usage in a template:  <script src="{{ asset('js/app.js') }}"></script>
    Unknown files degrade to an unversioned URL rather than raising — a
    missing asset should surface as a 404 in the network tab, not a 500 on
    the whole page.
    """
    rel = str(path).lstrip("/")
    full = BASE_DIR / "static" / rel
    try:
        st = full.stat()
    except OSError:
        return f"/static/{rel}"
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _ASSET_CACHE.get(rel)
    if hit and hit[0] == stamp:
        return hit[1]
    try:
        digest = hashlib.sha256(full.read_bytes()).hexdigest()[:10]
    except OSError:
        return f"/static/{rel}"
    url = f"/static/{rel}?v={digest}"
    _ASSET_CACHE[rel] = (stamp, url)
    return url


templates.env.globals["asset"] = asset

app.include_router(meta.router,       prefix="/api/meta")
app.include_router(skew.router,       prefix="/api/skew")
app.include_router(term.router,       prefix="/api/term")
app.include_router(historical.router, prefix="/api/historical")
app.include_router(concavity.router,  prefix="/api/convexity")
app.include_router(skew_slope.router, prefix="/api/skew_slope")
app.include_router(term_slope.router, prefix="/api/term_slope")
app.include_router(raw.router,        prefix="/api/raw")
app.include_router(heatmap.router,      prefix="/api/heatmap")
app.include_router(today.router,        prefix="/api/today")
app.include_router(ai_explorer.router,  prefix="/api/ai-explorer")
app.include_router(research.router,     prefix="/api/research")
app.include_router(research2.router,    prefix="/api/research2")
app.include_router(oi_signals.router,   prefix="/api/factor-signals")
app.include_router(oi_analysis.router,  prefix="/api/factor-analysis")
app.include_router(oi_portfolios.router, prefix="/api/factor-analysis")
app.include_router(backtest_iv.router,  prefix="/api/backtest-iv")
app.include_router(ticker_analysis.router, prefix="/api/ticker-analysis")
app.include_router(ticker_chain.router,    prefix="/api/ticker-analysis")
app.include_router(factor_trades.router,    prefix="/api/factor-trades")
app.include_router(equity_iv.router,        prefix="/api/equity-iv")
# Same prefix: the surface panels are the same page, split by which tables
# they read (equity_surface / equity_atm rather than the metric layer).
app.include_router(equity_iv_surface.router, prefix="/api/equity-iv")


@app.get("/today")
async def today_page(request: Request):
    return templates.TemplateResponse(request, "today.html")


@app.get("/heatmap")
async def heatmap_page(request: Request):
    return templates.TemplateResponse(request, "heatmap.html")


@app.get("/ai-explorer")
async def ai_explorer_page(request: Request):
    return templates.TemplateResponse(request, "ai_explorer.html")


@app.get("/research")
async def research_page(request: Request):
    return templates.TemplateResponse(request, "research.html")


@app.get("/research2")
async def research2_page(request: Request):
    return templates.TemplateResponse(request, "research2.html")


@app.get("/factor-analysis")
async def factor_analysis_page(request: Request):
    return templates.TemplateResponse(request, "oi_analysis.html")


@app.get("/factor-signals")
async def factor_signals_page(request: Request):
    return templates.TemplateResponse(request, "oi_signals.html")


@app.get("/factor-trades")
async def factor_trades_page(request: Request):
    return templates.TemplateResponse(request, "factor_trades.html")


@app.get("/ticker-analysis")
async def ticker_analysis_page(request: Request):
    return templates.TemplateResponse(request, "ticker_analysis.html")


@app.get("/equity-iv")
async def equity_iv_page(request: Request):
    return templates.TemplateResponse(request, "equity_iv.html")


@app.get("/backtest-iv-analysis")
async def backtest_iv_page(request: Request):
    return templates.TemplateResponse(request, "backtest_iv_analysis.html")


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")
