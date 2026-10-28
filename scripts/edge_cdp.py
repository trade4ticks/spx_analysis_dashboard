"""Drive a gate page in headless Edge IN REAL TIME, over the DevTools protocol.

Not a gate: the helper check_oo_split_ui and check_oo_offstate load their
pages with (2026-10-25).

WHY. The browser gates ran Edge with --dump-dom --virtual-time-budget: Edge
runs the page on virtual time and prints the DOM when that budget is spent.
Twice under the parallel suite check_oo_split_ui's page was printed partway
through -- Edge exited normally about five seconds in, at a step normal runs
reach with 70 of the 90 virtual seconds still unspent -- so the budget was
skipped through, not used up. A planted asynchronous wait did not reproduce
it and the cause was not found. This removes the mechanism rather than
tuning around it: the page runs on the real clock, and the gate waits for
the page's own report element, with a REAL timeout that names the last step
the page marked (the driver keeps `data-progress` / `data-trail` on <html>).

One page per Edge process, a private --user-data-dir each, serial: no
concurrency is added here (concurrency was already a suspect).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

REPORT_JS = "(() => { const r = document.getElementById('report'); return r ? r.textContent : null; })()"
WHERE_JS = ("(() => { const h = document.documentElement; return JSON.stringify({ progress: h.getAttribute('data-progress'), "
            "trail: h.getAttribute('data-trail'), alpine: !!window.Alpine, title: document.title }); })()")


class PageDidNotReport(RuntimeError):
    pass


def run_page(browser: str, page: Path, timeout_s: float = 300.0, window: str = "1600,1200") -> str:
    """Open `page` (a local file) and return the text of its #report element.
    Raises PageDidNotReport -- naming the last step the page marked, and its
    trail of steps -- when the report has not appeared within timeout_s."""
    from websockets.sync.client import connect

    prof = tempfile.mkdtemp(prefix="edge-cdp-")
    proc = subprocess.Popen(
        [browser, "--headless=new", "--disable-gpu", f"--window-size={window}", f"--user-data-dir={prof}",
         "--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        port = _devtools_port(Path(prof), proc)
        target = next(t for t in _json(f"http://127.0.0.1:{port}/json/list") if t.get("type") == "page")
        with connect(target["webSocketDebuggerUrl"], max_size=None, open_timeout=30) as ws:
            seq = [0]

            def call(method, params=None):
                seq[0] += 1
                ws.send(json.dumps({"id": seq[0], "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(ws.recv(timeout=60))
                    if msg.get("id") == seq[0]:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            def evaluate(expr):
                r = call("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                return (r.get("result") or {}).get("value")

            call("Page.navigate", {"url": page.resolve().as_uri()})
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                text = evaluate(REPORT_JS)
                if text is not None:
                    return text
                time.sleep(0.25)
            where = json.loads(evaluate(WHERE_JS) or "{}")
            raise PageDidNotReport(
                f"no report after {timeout_s:.0f}s of real time; last step "
                f"{where.get('progress') or '(none marked)'}; Alpine {'loaded' if where.get('alpine') else 'NOT loaded'}; "
                f"trail: {where.get('trail') or ''}")
    finally:
        proc.kill()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        shutil.rmtree(prof, ignore_errors=True)


def _devtools_port(prof: Path, proc: subprocess.Popen, wait_s: float = 30.0) -> int:
    """Edge writes the port it chose to <profile>/DevToolsActivePort."""
    f = prof / "DevToolsActivePort"
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"Edge exited at start ({proc.returncode}): {(proc.stderr.read() or b'')[-400:]!r}")
        if f.exists():
            # Edge may still hold the file open for writing: on Windows a read
            # then raises PermissionError (seen 2026-10-27, which killed a
            # re-record). Not ready yet -- poll again.
            try:
                first = f.read_text(encoding="utf-8").splitlines()
            except OSError:
                first = []
            if first and first[0].strip().isdigit():
                return int(first[0])
        time.sleep(0.05)
    raise RuntimeError("Edge did not open its DevTools port within 30s")


def _json(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))
