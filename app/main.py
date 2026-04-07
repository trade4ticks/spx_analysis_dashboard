from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db import init_pool, close_pool
from app.routers import meta, skew, term, historical, concavity

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
app.include_router(concavity.router,  prefix="/api/concavity")


@app.get("/")
async def index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))
