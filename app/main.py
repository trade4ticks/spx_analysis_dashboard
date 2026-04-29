from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_pool, close_pool
from app.routers import meta, skew, term, historical, concavity, skew_slope, term_slope, raw, heatmap, today, ai_explorer, research, research2

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


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))
