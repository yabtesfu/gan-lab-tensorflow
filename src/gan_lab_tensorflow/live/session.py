"""Background training session: runs the engine off the event loop.

TensorFlow-style blocking training and an asyncio web server do not mix on one
thread. So training runs in its own daemon thread that:

* steps the engine as fast as a target step-rate allows,
* snapshots telemetry into a single latest-frame slot at ~20 fps
  (a slow browser simply misses intermediate frames -- drop-to-latest
  coalescing, so the client never back-pressures training),
* reads its config under a lock at the top of every step, so steering
  messages take effect on the very next step without tearing anything down.

The web layer only ever reads ``latest_frame`` (an atomic reference swap) and
calls the thread-safe ``apply_control`` / ``play`` / ``pause`` / ``reset``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from .engine import LiveGan, LiveGanConfig, TelemetryFrame

_SPEED_STEPS_PER_SEC = {"slow": 60.0, "normal": 150.0, "fast": 400.0, "max": 5000.0}
_EMIT_PERIOD = 1.0 / 20.0  # 20 fps telemetry

# The deterministic "Demo Mode" preset: a fixed-seed, deliberately unstable
# configuration on the 3-mode mixture that reliably drives the generator into
# single-mode collapse (empirically coverage ~0.33) -- the setup for the
# collapse-and-rescue hero moment. Two honest fixes from here: RESET + switch
# to Wasserstein (prevention -> ~0.99 coverage), or turn on Instance Noise and
# drop the learning rate (a live, partial escape -- deep collapse never fully
# reverses, which is true to real GANs).
DEMO_COLLAPSE = LiveGanConfig(
    dataset="mixture",
    loss="vanilla",
    learning_rate=3e-3,
    ttur=False,
    d_steps=3,
    instance_noise=False,
    seed=7,
)


class TrainingSession:
    def __init__(self, config: LiveGanConfig | None = None):
        self._lock = threading.Lock()
        self._engine = LiveGan(config)
        self._running = False
        self._speed = "fast"
        self._stop = threading.Event()
        self.latest_frame: TelemetryFrame | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            self.latest_frame = self._engine.snapshot()
        self._thread = threading.Thread(target=self._loop, name="gan-observatory", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- controls (thread-safe) ---------------------------------------------
    def play(self) -> None:
        with self._lock:
            self._running = True

    def pause(self) -> None:
        with self._lock:
            self._running = False

    def toggle(self) -> bool:
        with self._lock:
            self._running = not self._running
            return self._running

    def reset(self) -> None:
        with self._lock:
            self._engine.reset()
            self.latest_frame = self._engine.snapshot()

    def set_speed(self, speed: str) -> None:
        if speed in _SPEED_STEPS_PER_SEC:
            with self._lock:
                self._speed = speed

    def load_demo(self) -> None:
        """Load the deterministic collapse preset, paused and ready to play."""
        with self._lock:
            self._engine.apply_config(replace(DEMO_COLLAPSE))
            self._running = False
            self.latest_frame = self._engine.snapshot()

    def apply_control(self, message: dict) -> None:
        """Handle a control message from the browser."""
        action = message.get("action")
        if action == "play":
            self.play()
        elif action == "pause":
            self.pause()
        elif action == "toggle":
            self.toggle()
        elif action == "reset":
            self.reset()
        elif action == "demo":
            self.load_demo()
        elif action == "speed":
            self.set_speed(str(message.get("value", "normal")))
        elif action == "config":
            self._apply_config_patch(message.get("value", {}))

    def _apply_config_patch(self, patch: dict) -> None:
        with self._lock:
            current = self._engine.config
            merged = LiveGanConfig(
                dataset=patch.get("dataset", current.dataset),
                loss=patch.get("loss", current.loss),
                noise_dim=int(patch.get("noise_dim", current.noise_dim)),
                hidden_units=int(patch.get("hidden_units", current.hidden_units)),
                batch_size=int(patch.get("batch_size", current.batch_size)),
                learning_rate=float(patch.get("learning_rate", current.learning_rate)),
                ttur=bool(patch.get("ttur", current.ttur)),
                d_steps=int(patch.get("d_steps", current.d_steps)),
                instance_noise=bool(patch.get("instance_noise", current.instance_noise)),
                seed=int(patch.get("seed", current.seed)),
            )
            rebuilt = self._engine.apply_config(merged)
            if rebuilt:
                self.latest_frame = self._engine.snapshot()

    def state(self) -> dict:
        with self._lock:
            cfg = self._engine.config
            return {
                "type": "state",
                "running": self._running,
                "speed": self._speed,
                "config": {
                    "dataset": cfg.dataset,
                    "loss": cfg.loss,
                    "noise_dim": cfg.noise_dim,
                    "hidden_units": cfg.hidden_units,
                    "batch_size": cfg.batch_size,
                    "learning_rate": cfg.learning_rate,
                    "ttur": cfg.ttur,
                    "d_steps": cfg.d_steps,
                    "instance_noise": cfg.instance_noise,
                    "seed": cfg.seed,
                },
            }

    # -- training loop -------------------------------------------------------
    def _loop(self) -> None:
        last_emit = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                running = self._running
                if running:
                    self._engine.train_step()
                target = _SPEED_STEPS_PER_SEC[self._speed]

            now = time.monotonic()
            if now - last_emit >= _EMIT_PERIOD:
                with self._lock:
                    self.latest_frame = self._engine.snapshot()
                last_emit = now

            if running:
                # Pace training so convergence is watchable rather than instant.
                time.sleep(max(0.0, 1.0 / target))
            else:
                time.sleep(0.03)
