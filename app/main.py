from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_pool, close_pool

# Raise multipart upload limit from 1MB to 200MB for backtest file uploads
try:
    from starlette.formparsers import MultiPartParser
    MultiPartParser.max_part_size = 200 * 1024 * 1024   # 200MB per file part
    MultiPartParser.spool_max_size = 200 * 1024 * 1024   # 200MB spool
except (ImportError, AttributeError):
    pass
from app.routers import meta, skew, term, historical, concavity, skew_slope, term_slope, raw, heatmap, today, ai_explorer, research, research2, oi_signals, oi_analysis, backtest_iv

BASE_DIR = Path(__file__).parent.parent  # project root


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="SPX IV Dashboard", lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

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
app.include_router(oi_signals.router,   prefix="/api/oi-signals")
app.include_router(oi_analysis.router,  prefix="/api/oi-analysis")
app.include_router(backtest_iv.router,  prefix="/api/backtest-iv")


@app.get("/today")
async def today_page():
    return FileResponse(str(BASE_DIR / "templates" / "today.html"))


@app.get("/heatmap")
async def heatmap_page():
    return FileResponse(str(BASE_DIR / "templates" / "heatmap.html"))


@app.get("/ai-explorer")
async def ai_explorer_page():
    return FileResponse(str(BASE_DIR / "templates" / "ai_explorer.html"))


@app.get("/research")
async def research_page():
    return FileResponse(str(BASE_DIR / "templates" / "research.html"))


@app.get("/research2")
async def research2_page():
    return FileResponse(str(BASE_DIR / "templates" / "research2.html"))


@app.get("/oi-analysis")
async def oi_analysis_page():
    return FileResponse(str(BASE_DIR / "templates" / "oi_analysis.html"))


@app.get("/oi-signals")
async def oi_signals_page():
    return FileResponse(str(BASE_DIR / "templates" / "oi_signals.html"))


@app.get("/backtest-iv-analysis")
async def backtest_iv_page():
    return FileResponse(str(BASE_DIR / "templates" / "backtest_iv_analysis.html"))


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))
