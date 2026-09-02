"""Equities Live — its own process, its own port.

Deliberately not part of run.py. If the tape service crashes it must not take
the dashboards down with it, and a shared uvicorn would make that impossible
to guarantee.

    python run_live.py            # LIVE_PORT, default 8001
"""
import os

import uvicorn

from live import config

if __name__ == "__main__":
    for p in config.problems():
        print(f"configuration: {p}")
    print(f"feed: {config.FEED} -> {config.feed_url()}")
    print(f"caps: {config.MAX_SYMBOLS} symbols, "
          f"{config.MAX_WINDOW_S}s window, {config.MAX_CLIENTS} clients")
    uvicorn.run("live.main:app", host=config.HOST, port=config.PORT,
                reload=False)
