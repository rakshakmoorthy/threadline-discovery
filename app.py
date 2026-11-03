"""Threadline Discovery web app (FastAPI).

Serves the dashboard, the latest saved results, and a protected 're-run' endpoint.
Run locally:   uvicorn app:app --reload
Render start:  uvicorn app:app --host 0.0.0.0 --port $PORT
"""
import json
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline

BASE = Path(__file__).parent
(BASE / "static" / "concepts").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Threadline Discovery")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")

LABELS = {
    "connect.mayoclinic.org": "Mayo Clinic Connect",
    "community.breastcancer.org": "Breastcancer.org",
    "forum.breastcancernow.org": "Breast Cancer Now",
    "community.macmillan.org.uk": "Macmillan",
    "onlinenetwork.bcna.org.au": "BCNA",
    "csn.cancer.org": "ACS CSN",
}


def build_index():
    """Write static/signals_index.json: signal ids grouped by source thread.
    It holds no comment text, so it is safe to publish and powers the signal map."""
    src = BASE / "data" / "signals_public.json"
    if not src.exists():
        return
    try:
        signals = json.loads(src.read_text(encoding="utf-8"))
        threads = {}
        for s in signals:
            url = s.get("url", "unknown")
            host = urlparse(url).netloc.lower()
            t = threads.setdefault(url, {"url": url, "label": LABELS.get(host, host or "other"), "ids": []})
            t["ids"].append(s["id"])
        out = {"signal_count": len(signals), "threads": list(threads.values())}
        (BASE / "static" / "signals_index.json").write_text(json.dumps(out), encoding="utf-8")
    except Exception as e:
        print("could not build signals index:", e)


build_index()

COOLDOWN_SECONDS = 300  # protects API credits from repeated clicks
_lock = threading.Lock()
_last_start = 0.0
_state = {"status": "idle", "stage": "", "message": "", "pct": 0, "error": None, "events": []}


def _progress(stage, message, pct, event=None):
    _state.update(stage=stage, message=message, pct=pct)
    if event is not None:
        _state["events"].append(event)


def _worker():
    try:
        pipeline.run(progress=_progress)
        build_index()
        _state.update(status="done", stage="done", pct=100, error=None)
    except Exception as e:  # keep the last good results.json in place
        _state.update(status="error", error=str(e)[:300])


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/api/results")
def results():
    path = BASE / "results.json"
    if not path.exists():
        return JSONResponse({"opportunities": [], "rejected": []}, status_code=404)
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/status")
def status():
    wait = max(0, int(COOLDOWN_SECONDS - (time.time() - _last_start)))
    return {**_state, "cooldown": wait if _state["status"] != "running" else 0}


@app.post("/api/run")
def run_analysis():
    global _last_start
    with _lock:
        if _state["status"] == "running":
            return JSONResponse({"error": "A run is already in progress."}, status_code=409)
        wait = int(COOLDOWN_SECONDS - (time.time() - _last_start))
        if wait > 0:
            return JSONResponse({"error": f"Please wait {wait}s before running again.",
                                 "cooldown": wait}, status_code=429)
        _last_start = time.time()
        _state.update(status="running", stage="propose", message="Starting", pct=2, error=None, events=[])
    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True}
