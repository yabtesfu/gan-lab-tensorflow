"""A dependency-light, steerable 2D GAN you can watch train in real time.

This is intentionally *not* the TensorFlow trainer. TensorFlow cannot be
watched converging in a browser at interactive frame rates on CPU without a
heavyweight install, and the whole point of the observatory is that the live
2D case is small enough to see. So the live backend is a hand-written MLP GAN
with manual forward/backprop and Adam.

What it still shares with the rest of the package (genuine reuse, not a fork):

* Target distributions come from :mod:`gan_lab_tensorflow.data`
  (``sample_curve`` / ``sample_mixture``), so "what real looks like" is
  identical to the offline lab.
* Live quality metrics come from :mod:`gan_lab_tensorflow.evaluation`
  (RBF-MMD and nearest-neighbour precision/recall), so the numbers on screen
  are the same estimators the offline lab reports.
* The discriminator mirrors ``models.build_mlp_discriminator``: hidden
  LeakyReLU stack -> a linear 2D ``feature_plane`` bottleneck -> a scalar
  logit. That bottleneck is streamed to the browser as its own panel.

Everything is in a normalised space (per-axis zero-mean/unit-std, estimated
once from the target) so a single Adam configuration trains every
distribution; points are de-normalised back to data coordinates before they
leave the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..data import Point, sample_curve, sample_mixture, sample_ring, sine_y
from ..evaluation import maximum_mean_discrepancy, nearest_neighbor_precision

Dataset = Literal["quadratic", "sine", "mixture", "ring"]
LossKind = Literal["vanilla", "wasserstein"]
DATASETS = ("quadratic", "sine", "mixture", "ring")

_LEAKY_ALPHA = 0.2
_WEIGHT_CLIP = 0.04  # Lipschitz clip for the weight-clipped WGAN path


def _sample_real(dataset: Dataset, count: int, *, seed: int | None) -> list[Point]:
    """Draw real points from the package's own distributions."""
    if dataset == "quadratic":
        return sample_curve(count, noise=0.6, seed=seed)
    if dataset == "sine":
        return sample_curve(count, fn=sine_y, noise=0.15, seed=seed)
    if dataset == "mixture":
        return sample_mixture(count, seed=seed)
    if dataset == "ring":
        return sample_ring(count, seed=seed)
    raise ValueError(f"unknown dataset {dataset!r}")


def _leaky(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.0, x, _LEAKY_ALPHA * x)


def _dleaky(z: np.ndarray) -> np.ndarray:
    return np.where(z > 0.0, 1.0, _LEAKY_ALPHA)


class _MLP:
    """A tiny fully-connected net with manual forward/backprop and Adam.

    Layers are ``(units, activation)`` pairs; ``activation`` is ``"leaky"`` or
    ``"linear"``. Backprop returns both parameter gradients and the gradient
    with respect to the input, which is what lets the generator learn through
    the (frozen) discriminator.
    """

    def __init__(self, in_dim: int, spec: list[tuple[int, str]], rng: np.random.Generator):
        self.spec = spec
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        prev = in_dim
        for units, _act in spec:
            # He-style init keeps activations alive through LeakyReLU stacks.
            scale = np.sqrt(2.0 / prev)
            self.weights.append(rng.standard_normal((prev, units)).astype(np.float64) * scale)
            self.biases.append(np.zeros(units, dtype=np.float64))
            prev = units
        # Adam moment buffers, one per parameter tensor.
        self._mw = [np.zeros_like(w) for w in self.weights]
        self._vw = [np.zeros_like(w) for w in self.weights]
        self._mb = [np.zeros_like(b) for b in self.biases]
        self._vb = [np.zeros_like(b) for b in self.biases]
        self._t = 0

    @classmethod
    def from_state(cls, spec, weights, biases) -> "_MLP":
        """Rebuild an inference-only net from saved arrays (no optimizer state)."""
        obj = cls.__new__(cls)
        obj.spec = [tuple(s) for s in spec]
        obj.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        obj.biases = [np.asarray(b, dtype=np.float64) for b in biases]
        return obj

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        cache: list[tuple[np.ndarray, np.ndarray]] = []
        a = x
        for i, (_units, act) in enumerate(self.spec):
            z = a @ self.weights[i] + self.biases[i]
            cache.append((a, z))
            a = _leaky(z) if act == "leaky" else z
        return a, cache

    def backward(
        self, dout: np.ndarray, cache: list[tuple[np.ndarray, np.ndarray]]
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        dws: list[np.ndarray] = [np.zeros_like(w) for w in self.weights]
        dbs: list[np.ndarray] = [np.zeros_like(b) for b in self.biases]
        da = dout
        for i in reversed(range(len(self.spec))):
            a_in, z = cache[i]
            act = self.spec[i][1]
            dz = da if act == "linear" else da * _dleaky(z)
            dws[i] = a_in.T @ dz
            dbs[i] = dz.sum(axis=0)
            da = dz @ self.weights[i].T
        return dws, dbs, da

    def adam_step(
        self,
        dws: list[np.ndarray],
        dbs: list[np.ndarray],
        lr: float,
        *,
        b1: float = 0.5,
        b2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self._t += 1
        bc1 = 1.0 - b1 ** self._t
        bc2 = 1.0 - b2 ** self._t
        for i in range(len(self.weights)):
            self._mw[i] = b1 * self._mw[i] + (1 - b1) * dws[i]
            self._vw[i] = b2 * self._vw[i] + (1 - b2) * (dws[i] ** 2)
            self.weights[i] -= lr * (self._mw[i] / bc1) / (np.sqrt(self._vw[i] / bc2) + eps)
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * dbs[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * (dbs[i] ** 2)
            self.biases[i] -= lr * (self._mb[i] / bc1) / (np.sqrt(self._vb[i] / bc2) + eps)

    def clip_weights(self, c: float) -> None:
        for w in self.weights:
            np.clip(w, -c, c, out=w)
        for b in self.biases:
            np.clip(b, -c, c, out=b)


@dataclass
class LiveGanConfig:
    """Everything a viewer can steer while the run is live."""

    dataset: Dataset = "mixture"
    loss: LossKind = "wasserstein"  # a clean multimodal win on first Train
    noise_dim: int = 16
    hidden_units: int = 64
    batch_size: int = 128
    learning_rate: float = 8e-4
    ttur: bool = False           # discriminator LR = 4x generator LR
    d_steps: int = 1             # discriminator updates per generator update
    instance_noise: bool = False  # decaying Gaussian noise on D inputs (a stabiliser)
    seed: int = 42

    def clamp(self) -> "LiveGanConfig":
        self.noise_dim = int(np.clip(self.noise_dim, 1, 64))
        self.hidden_units = int(np.clip(self.hidden_units, 8, 256))
        self.batch_size = int(np.clip(self.batch_size, 16, 512))
        self.learning_rate = float(np.clip(self.learning_rate, 1e-5, 1e-1))
        self.d_steps = int(np.clip(self.d_steps, 1, 5))
        if self.loss not in ("vanilla", "wasserstein"):
            raise ValueError("loss must be 'vanilla' or 'wasserstein'")
        if self.dataset not in DATASETS:
            raise ValueError("dataset must be one of: " + ", ".join(DATASETS))
        return self


@dataclass
class TelemetryFrame:
    """One snapshot pushed to the browser."""

    step: int
    gen_loss: float
    disc_loss: float
    mmd: float
    coverage: float          # recall: fraction of real modes the generator reaches
    precision: float         # fraction of generated points that look real
    collapsed: bool
    real_points: list[list[float]]
    fake_points: list[list[float]]
    feature_real: list[list[float]]
    feature_fake: list[list[float]]
    grid: dict               # decision-boundary heatmap in data coordinates
    extent: list[float]      # [xmin, xmax, ymin, ymax] in data coordinates
    loss: str
    dataset: str

    def to_dict(self) -> dict:
        return {
            "type": "frame",
            "step": self.step,
            "genLoss": round(self.gen_loss, 4),
            "discLoss": round(self.disc_loss, 4),
            "mmd": round(self.mmd, 4),
            "coverage": round(self.coverage, 4),
            "precision": round(self.precision, 4),
            "collapsed": self.collapsed,
            "real": self.real_points,
            "fake": self.fake_points,
            "featReal": self.feature_real,
            "featFake": self.feature_fake,
            "grid": self.grid,
            "extent": self.extent,
            "loss": self.loss,
            "dataset": self.dataset,
        }


class LiveGan:
    """A steerable 2D GAN whose internals can be inspected every step."""

    _GRID = 32           # decision-boundary resolution (GRID x GRID)
    _VIZ_POINTS = 160    # points streamed for the scatter panels
    _METRIC_POINTS = 96
    _METRIC_REFRESH = 15  # min training steps between (expensive) metric recomputes

    def __init__(self, config: LiveGanConfig | None = None):
        self.config = (config or LiveGanConfig()).clamp()
        self.step = 0
        self._collapse_streak = 0
        self._build()

    # -- setup ---------------------------------------------------------------
    def _build(self) -> None:
        cfg = self.config
        # Two independent RNG streams: one drives training (weight init, latent
        # noise, batch draws), the other only produces the fixed viz latents.
        # Keeping them separate means snapshotting -- which happens at a
        # timing-dependent cadence -- never perturbs the training trajectory,
        # so a given (seed, config, step-count) is reproducible.
        self._rng = np.random.default_rng(cfg.seed)
        viz_rng = np.random.default_rng(cfg.seed + 12345)

        # A fixed real pool drawn once from the package's own distributions.
        pool = np.asarray(_sample_real(cfg.dataset, 8192, seed=cfg.seed), dtype=np.float64)
        self._mean = pool.mean(axis=0)
        self._std = pool.std(axis=0) + 1e-8
        self._real_pool = self._normalise(pool)
        self._pool_size = len(pool)
        # A stationary real reference for the scatter/metrics (the blue cloud
        # should hold still while the generator chases it).
        self._real_ref = self._real_pool[: self._VIZ_POINTS]

        lo = pool.min(axis=0)
        hi = pool.max(axis=0)
        pad = 0.15 * (hi - lo + 1e-6)
        self._extent = [lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]]
        self._make_grid()

        h = cfg.hidden_units
        self._gen = _MLP(cfg.noise_dim, [(h, "leaky"), (h, "leaky"), (2, "linear")], self._rng)
        # Discriminator mirrors models.build_mlp_discriminator: hidden stack,
        # linear 2D feature_plane, then the scalar logit.
        self._disc = _MLP(2, [(h, "leaky"), (h, "leaky"), (2, "linear"), (1, "linear")], self._rng)
        # Fixed latents so the same generated points are tracked every frame.
        self._viz_noise = viz_rng.uniform(-1.0, 1.0, size=(self._VIZ_POINTS, cfg.noise_dim))
        # Metric cache: the O(n^2) estimators are throttled off the 20 fps path.
        self._metric_cache: tuple[float, float, float] | None = None
        self._metric_step = -10 ** 9

    def _make_grid(self) -> None:
        xmin, xmax, ymin, ymax = self._extent
        gx = np.linspace(xmin, xmax, self._GRID)
        gy = np.linspace(ymin, ymax, self._GRID)
        mesh_x, mesh_y = np.meshgrid(gx, gy)
        pts = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=1)
        self._grid_pts_norm = self._normalise(pts)

    def _normalise(self, pts: np.ndarray) -> np.ndarray:
        return (pts - self._mean) / self._std

    def _denormalise(self, pts: np.ndarray) -> np.ndarray:
        return pts * self._std + self._mean

    # -- steering ------------------------------------------------------------
    def apply_config(self, new: LiveGanConfig) -> bool:
        """Apply steering. Returns True if the model had to be rebuilt.

        Hyperparameters that change tensor shapes (dataset/width/noise) force a
        rebuild; the rest (loss, LR, TTUR, d_steps, noise) take effect on the
        very next step without disturbing the weights.
        """
        new = new.clamp()
        old = self.config
        structural = (
            new.dataset != old.dataset
            or new.hidden_units != old.hidden_units
            or new.noise_dim != old.noise_dim
            or new.seed != old.seed
        )
        self.config = new
        if structural:
            self.step = 0
            self._collapse_streak = 0
            self._build()
            return True
        return False

    def reset(self) -> None:
        self.step = 0
        self._collapse_streak = 0
        self._build()

    # -- serialization -------------------------------------------------------
    def export_state(self) -> dict:
        """A self-contained snapshot of the *generator* for serving.

        Captures the generator weights plus the normaliser and display extent,
        so a saved run can be reloaded and sampled from without retraining.
        """
        return {
            "noise_dim": int(self.config.noise_dim),
            "spec": [[int(u), a] for (u, a) in self._gen.spec],
            "weights": [w.tolist() for w in self._gen.weights],
            "biases": [b.tolist() for b in self._gen.biases],
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "extent": [float(v) for v in self._extent],
            "dataset": self.config.dataset,
            "loss": self.config.loss,
            "seed": int(self.config.seed),
        }

    # -- training ------------------------------------------------------------
    def _noise(self, n: int) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(n, self.config.noise_dim))

    def _real_batch(self, n: int) -> np.ndarray:
        # Deterministic mini-batches: draw indices from the fixed real pool with
        # the (seeded) training RNG.
        idx = self._rng.integers(0, self._pool_size, size=n)
        return self._real_pool[idx]

    def _maybe_instance_noise(self, x: np.ndarray) -> np.ndarray:
        if not self.config.instance_noise:
            return x
        sigma = 0.3 * np.exp(-self.step / 1500.0)  # anneal toward zero
        return x + self._rng.normal(0.0, sigma, size=x.shape)

    def train_step(self) -> None:
        cfg = self.config
        gen_lr = cfg.learning_rate
        disc_lr = cfg.learning_rate * (4.0 if cfg.ttur else 1.0)
        n = cfg.batch_size

        # --- discriminator updates (possibly several per generator update) ---
        for _ in range(cfg.d_steps):
            real = self._maybe_instance_noise(self._real_batch(n))
            fake = self._maybe_instance_noise(self._gen.forward(self._noise(n))[0])
            real_logit, real_cache = self._disc.forward(real)
            fake_logit, fake_cache = self._disc.forward(fake)
            if cfg.loss == "wasserstein":
                dreal = np.full_like(real_logit, -1.0 / n)
                dfake = np.full_like(fake_logit, 1.0 / n)
            else:  # vanilla / non-saturating BCE-with-logits
                dreal = (_sigmoid(real_logit) - 1.0) / n
                dfake = _sigmoid(fake_logit) / n
            dwr, dbr, _ = self._disc.backward(dreal, real_cache)
            dwf, dbf, _ = self._disc.backward(dfake, fake_cache)
            dws = [a + b for a, b in zip(dwr, dwf)]
            dbs = [a + b for a, b in zip(dbr, dbf)]
            self._disc.adam_step(dws, dbs, disc_lr)
            if cfg.loss == "wasserstein":
                self._disc.clip_weights(_WEIGHT_CLIP)

        # --- generator update -------------------------------------------------
        z = self._noise(n)
        fake, gen_cache = self._gen.forward(z)
        fake_in = self._maybe_instance_noise(fake)
        fake_logit, fake_cache = self._disc.forward(fake_in)
        if cfg.loss == "wasserstein":
            dlogit = np.full_like(fake_logit, -1.0 / n)
        else:  # non-saturating: maximise log D(G(z))
            dlogit = (_sigmoid(fake_logit) - 1.0) / n
        # Backprop through the frozen discriminator to its input, then into G.
        _, _, dfake = self._disc.backward(dlogit, fake_cache)
        gdw, gdb, _ = self._gen.backward(dfake, gen_cache)
        self._gen.adam_step(gdw, gdb, gen_lr)

        self.step += 1

    # -- telemetry -----------------------------------------------------------
    def _boundary_grid(self) -> dict:
        logits, _ = self._disc.forward(self._grid_pts_norm)
        prob = _sigmoid(logits).reshape(self._GRID, self._GRID)
        return {"n": self._GRID, "values": [round(float(v), 3) for v in prob.ravel()]}

    def snapshot(self) -> TelemetryFrame:
        cfg = self.config
        # Fixed latents + stationary real reference => snapshotting is a pure
        # read of the current weights and never touches the training RNG.
        fake_norm, _ = self._gen.forward(self._viz_noise)
        real_norm = self._real_ref
        fake_disp = self._denormalise(fake_norm)
        real_disp = self._denormalise(real_norm)

        # Discriminator feature-plane coordinates for the same points.
        _, real_cache = self._disc.forward(real_norm)
        _, fake_cache = self._disc.forward(fake_norm)
        feat_real = real_cache[-1][0]  # input to the logit layer == feature_plane
        feat_fake = fake_cache[-1][0]

        # Losses (reported, not the raw per-step gradient scalar).
        gen_loss, disc_loss = self._report_losses(real_norm, fake_norm)

        mmd, coverage, precision = self._metrics(real_norm, fake_norm)
        collapsed = self._update_collapse(coverage, fake_norm)

        return TelemetryFrame(
            step=self.step,
            gen_loss=gen_loss,
            disc_loss=disc_loss,
            mmd=mmd,
            coverage=coverage,
            precision=precision,
            collapsed=collapsed,
            real_points=_round_points(real_disp),
            fake_points=_round_points(fake_disp),
            feature_real=_round_points(feat_real),
            feature_fake=_round_points(feat_fake),
            grid=self._boundary_grid(),
            extent=[round(float(v), 3) for v in self._extent],
            loss=cfg.loss,
            dataset=cfg.dataset,
        )

    def _metrics(self, real_norm: np.ndarray, fake_norm: np.ndarray) -> tuple[float, float, float]:
        """RBF-MMD, coverage (recall) and precision from the package estimators.

        These are O(n^2) pure Python, so they are recomputed only after real
        training progress (every ``_METRIC_REFRESH`` steps) and cached in
        between -- keeping the 20 fps telemetry path cheap. A fresh snapshot
        right after training always reflects the latest weights.
        """
        if self._metric_cache is not None and (self.step - self._metric_step) < self._METRIC_REFRESH:
            return self._metric_cache
        m = self._METRIC_POINTS
        real_pts = [tuple(p) for p in real_norm[:m]]
        fake_pts = [tuple(p) for p in fake_norm[:m]]
        mmd = maximum_mean_discrepancy(real_pts, fake_pts, sigma=1.0)
        precision = nearest_neighbor_precision(real_pts, fake_pts, radius=0.35)
        coverage = nearest_neighbor_precision(fake_pts, real_pts, radius=0.35)  # recall
        self._metric_cache = (mmd, coverage, precision)
        self._metric_step = self.step
        return self._metric_cache

    def _report_losses(self, real_norm: np.ndarray, fake_norm: np.ndarray) -> tuple[float, float]:
        real_logit, _ = self._disc.forward(real_norm)
        fake_logit, _ = self._disc.forward(fake_norm)
        if self.config.loss == "wasserstein":
            disc = float(np.mean(fake_logit) - np.mean(real_logit))
            gen = float(-np.mean(fake_logit))
        else:
            disc = float(np.mean(_softplus(-real_logit)) + np.mean(_softplus(fake_logit)))
            gen = float(np.mean(_softplus(-fake_logit)))
        return gen, disc

    def _update_collapse(self, coverage: float, fake_norm: np.ndarray) -> bool:
        # Coverage (recall) is the scale-free mode-collapse signal: the fraction
        # of the real distribution the generator still reaches. Healthy runs sit
        # at 0.7-0.99; abandoning modes drops it well below 0.4 on both the
        # compact mixture and the wide 8-Gaussian ring. Require a short streak so
        # transient early-training dips do not trip the alarm.
        bad = coverage < 0.4
        self._collapse_streak = self._collapse_streak + 1 if bad else 0
        return self._collapse_streak >= 4


def sample_generator(state: dict, count: int, *, seed: int | None = None) -> dict:
    """Generate fresh samples from a saved generator state (see export_state).

    This is what backs the ``/generate`` serving endpoint: load a run, draw
    ``count`` samples in data coordinates. Pure inference -- no discriminator,
    no optimizer, no training.
    """
    count = int(max(1, min(count, 5000)))
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, size=(count, int(state["noise_dim"])))
    gen = _MLP.from_state(state["spec"], state["weights"], state["biases"])
    out, _ = gen.forward(z)
    disp = out * np.asarray(state["std"]) + np.asarray(state["mean"])
    return {
        "points": _round_points(disp),
        "extent": [float(v) for v in state["extent"]],
        "dataset": state.get("dataset"),
        "loss": state.get("loss"),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (np.tanh(0.5 * x) + 1.0)


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def _round_points(arr: np.ndarray, ndigits: int = 3) -> list[list[float]]:
    return [[round(float(a), ndigits), round(float(b), ndigits)] for a, b in arr]
