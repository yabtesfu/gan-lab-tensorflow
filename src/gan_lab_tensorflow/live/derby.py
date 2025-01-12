"""The A/B derby: two GANs racing on the same target from the same seed.

Both engines start *identical* -- same dataset, same seed, so same real pool
and same weight initialisation -- and differ only in the loss/stabilisers.
Stepped in lockstep on one thread and streamed as paired frames, they make the
point no static plot can: watch the vanilla configuration collapse onto a
single mode while the Wasserstein one covers them all, side by side, on the
same clock.

Reuses the Phase 1 engine and the same background-thread + latest-frame
coalescing pattern as :class:`~gan_lab_tensorflow.live.session.TrainingSession`.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

from .engine import LiveGan, LiveGanConfig
from .session import _EMIT_PERIOD, _SPEED_STEPS_PER_SEC

# The two contestants. Same dataset + seed are injected at construction; only
# the loss and its stabilisers differ, so the race is a fair controlled test.
LEFT_CONFIG = LiveGanConfig(loss="vanilla", learning_rate=3e-3, d_steps=3)
RIGHT_CONFIG = LiveGanConfig(loss="wasserstein", learning_rate=8e-4, d_steps=1)

_DERBY_DATASETS = ("mixture", "ring")


class DerbySession:
    def __init__(self, dataset: str = "mixture", seed: int = 7):
        self._lock = threading.Lock()
        self._dataset = dataset if dataset in _DERBY_DATASETS else "mixture"
        self._seed = int(seed)
        self._running = False
        self._speed = "fast"
        self._stop = threading.Event()
        self.latest_frame: dict | None = None
        self._thread: threading.Thread | None = None
        self._build()

    def _build(self) -> None:
        self._left = LiveGan(replace(LEFT_CONFIG, dataset=self._dataset, seed=self._seed))
        self._right = LiveGan(replace(RIGHT_CONFIG, dataset=self._dataset, seed=self._seed))

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        with self._lock:
            self.latest_frame = self._combined()
        self._thread = threading.Thread(target=self._loop, name="gan-derby", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- controls ------------------------------------------------------------
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
            self._build()
            self._running = False
            self.latest_frame = self._combined()

    def set_dataset(self, dataset: str) -> None:
        if dataset in _DERBY_DATASETS:
            with self._lock:
                self._dataset = dataset
                self._build()
                self._running = False
                self.latest_frame = self._combined()

    def set_speed(self, speed: str) -> None:
        if speed in _SPEED_STEPS_PER_SEC:
            with self._lock:
                self._speed = speed

    def apply_control(self, message: dict) -> None:
        action = message.get("action")
        if action == "play":
            self.play()
        elif action == "pause":
            self.pause()
        elif action == "toggle":
            self.toggle()
        elif action == "reset":
            self.reset()
        elif action == "dataset":
            self.set_dataset(str(message.get("value", "mixture")))
        elif action == "speed":
            self.set_speed(str(message.get("value", "fast")))

    def state(self) -> dict:
        with self._lock:
            return {
                "type": "state",
                "running": self._running,
                "speed": self._speed,
                "dataset": self._dataset,
                "seed": self._seed,
                "left": {"label": "Vanilla GAN", "loss": self._left.config.loss},
                "right": {"label": "Wasserstein", "loss": self._right.config.loss},
            }

    # -- telemetry -----------------------------------------------------------
    @staticmethod
    def _side(frame) -> dict:
        return {
            "fake": frame.fake_points,
            "grid": frame.grid["values"],
            "mmd": round(frame.mmd, 4),
            "coverage": round(frame.coverage, 4),
            "collapsed": frame.collapsed,
            "gen": round(frame.gen_loss, 3),
            "disc": round(frame.disc_loss, 3),
            "loss": frame.loss,
        }

    def _combined(self) -> dict:
        lf = self._left.snapshot()
        rf = self._right.snapshot()
        # Both engines share dataset+seed, so real reference, extent and grid
        # coordinates are identical -- send them once.
        return {
            "type": "derby",
            "step": lf.step,
            "extent": lf.extent,
            "gridN": lf.grid["n"],
            "real": lf.real_points,
            "left": self._side(lf),
            "right": self._side(rf),
        }

    # -- loop ----------------------------------------------------------------
    def _loop(self) -> None:
        last_emit = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                running = self._running
                if running:
                    self._left.train_step()
                    self._right.train_step()
                target = _SPEED_STEPS_PER_SEC[self._speed]

            now = time.monotonic()
            if now - last_emit >= _EMIT_PERIOD:
                with self._lock:
                    self.latest_frame = self._combined()
                last_emit = now

            time.sleep(max(0.0, 1.0 / target) if running else 0.03)
