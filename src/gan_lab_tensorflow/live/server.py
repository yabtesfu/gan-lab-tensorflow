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
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .session import _EMIT_PERIOD, TrainingSession

_STATIC = Path(__file__).parent / "static"

app = FastAPI(title="GAN Observatory")
app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

# Single-session guard: one live training run at a time.
_session_busy = asyncio.Lock()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(_STATIC / "index.html"))


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    if _session_busy.locked():
        await ws.send_json({"type": "error", "message": "A training session is already running. Try again shortly."})
        await ws.close()
        return

    async with _session_busy:
        session = TrainingSession()
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
