"""Real-time GAN training observatory.

This package turns the offline lab into a live, steerable web experience:
a training loop that streams telemetry to the browser and reacts to control
messages mid-run.

The default live backend (:mod:`.engine`) is a dependency-light NumPy GAN so
the 2D theatre runs anywhere without TensorFlow. It deliberately reuses the
package's real target distributions (:mod:`gan_lab_tensorflow.data`) and
distribution metrics (:mod:`gan_lab_tensorflow.evaluation`). The TensorFlow
models remain the image backend for later phases.
"""

from .engine import LiveGan, LiveGanConfig, TelemetryFrame

__all__ = ["LiveGan", "LiveGanConfig", "TelemetryFrame"]
