"""FastAPI + WebSocket front door for the GAN Observatory.

One WebSocket carries the whole experience: telemetry frames stream out at
~20 fps from the background training thread, and steering messages (play,
pause, hyperparameter changes, the demo preset) stream back in. Frames are
JSON with compact rounded arrays -- small enough for localhost/free-tier
hosting; a binary msgpack protocol is the obvious later optimisation.

A single-session lock keeps one live training run at a time, so a public URL
cannot be driven into the ground by concurrent visitors.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .engine import sample_generator
from .registry import RunRegistry
from .session import _EMIT_PERIOD, TrainingSession

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="GAN Observatory")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# One registry of reproducible runs for the whole process.
registry = RunRegistry(os.environ.get("RUNS_DB", "runs.db"))

# Single-session guard: one live training run at a time.
_session_busy = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/runs")
async def api_runs() -> dict:
    """List saved runs (newest first) for the run registry panel."""
    return {"runs": registry.list()}


@app.get("/api/runs/{run_id}")
async def api_run(run_id: int) -> dict:
    """A single run including its streamed metric history."""
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


@app.get("/api/runs/{run_id}/sample")
async def api_sample(run_id: int, count: int = 240, seed: Optional[int] = None) -> dict:
    """Serve fresh samples from a saved generator -- the model-serving endpoint.

    Pure inference: reloads the run's generator and draws ``count`` new points.
    """
    run = registry.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    result = sample_generator(run["state"], count, seed=seed)
    result["runId"] = run_id
    return result


@app.delete("/api/runs/{run_id}")
async def api_delete(run_id: int) -> dict:
    return {"deleted": registry.delete(run_id)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    if _session_busy.locked():
        await ws.send_json({"type": "error", "message": "A training session is already running. Try again shortly."})
        await ws.close()
        return

    async with _session_busy:
        session = TrainingSession(registry=registry)
        session.start()
        send_lock = asyncio.Lock()

        async def safe_send(payload: dict) -> None:
            # Two coroutines share one socket; serialise sends so frames and
            # control replies never interleave on the wire.
            async with send_lock:
                await ws.send_json(payload)

        try:
            await safe_send(session.state())
            if session.latest_frame is not None:
                await safe_send(session.latest_frame.to_dict())
            await asyncio.gather(
                _sender(safe_send, session),
                _receiver(ws, safe_send, session),
            )
        except WebSocketDisconnect:
            pass
        finally:
            session.stop()


async def _sender(safe_send, session: TrainingSession) -> None:
    """Push the latest telemetry frame at a fixed cadence."""
    last_step = -1
    while True:
        await asyncio.sleep(_EMIT_PERIOD)
        frame = session.latest_frame
        if frame is None:
            continue
        # Skip re-sending an identical paused frame, but always refresh on step change.
        if frame.step == last_step:
            continue
        last_step = frame.step
        await safe_send(frame.to_dict())


async def _receiver(ws: WebSocket, safe_send, session: TrainingSession) -> None:
    """Apply steering messages and echo state back so the UI stays in sync."""
    while True:
        message = await ws.receive_json()
        if message.get("action") == "save":
            # Persist the current run, then hand back the refreshed run list.
            run_id = await asyncio.to_thread(session.save_run)
            await safe_send({"type": "saved", "id": run_id})
            await safe_send({"type": "runs", "runs": registry.list()})
            continue
        session.apply_control(message)
        await safe_send(session.state())
        # Structural steering (dataset/width/demo/reset) refreshes the frame
        # immediately even while paused, so the scatter updates on click.
        frame = session.latest_frame
        if frame is not None:
            await safe_send(frame.to_dict())


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import os

    run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
