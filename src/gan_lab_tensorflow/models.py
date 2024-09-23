from __future__ import annotations

from .config import ModelConfig


def _tf():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model construction. "
            "Install with `pip install tensorflow` in a supported Python environment."
        ) from exc
    return tf


def build_mlp_generator(config: ModelConfig):
    tf = _tf()
    model = tf.keras.Sequential(name="generator")
    model.add(tf.keras.layers.Input(shape=(config.noise_dim,)))
    for units in config.hidden_units:
        model.add(tf.keras.layers.Dense(units))
        model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dense(config.data_dim))
    return model


def build_mlp_discriminator(config: ModelConfig, *, return_features: bool = False):
    tf = _tf()
    inputs = tf.keras.Input(shape=(config.data_dim,), name="sample")
    x = inputs
    for units in config.hidden_units:
        x = tf.keras.layers.Dense(units)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
    features = tf.keras.layers.Dense(2, activation="linear", name="feature_plane")(x)
    logits = tf.keras.layers.Dense(1, name="logits")(features)
    if return_features:
        return tf.keras.Model(inputs, [logits, features], name="discriminator")
    return tf.keras.Model(inputs, logits, name="discriminator")


def build_dcgan_generator(config: ModelConfig):
    tf = _tf()
    height, width, channels = config.image_shape
    if (height, width) != (28, 28):
        raise ValueError("DCGAN helper currently expects 28x28 images")
    model = tf.keras.Sequential(name="dcgan_generator")
    model.add(tf.keras.layers.Input(shape=(config.noise_dim,)))
    model.add(tf.keras.layers.Dense(7 * 7 * 128, use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Reshape((7, 7, 128)))
    model.add(tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding="same", use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Conv2DTranspose(channels, 5, strides=2, padding="same", activation="tanh"))
    return model


def build_dcgan_discriminator(config: ModelConfig):
    tf = _tf()
    model = tf.keras.Sequential(name="dcgan_discriminator")
    model.add(tf.keras.layers.Input(shape=config.image_shape))
    model.add(tf.keras.layers.Conv2D(64, 5, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Conv2D(128, 5, strides=2, padding="same"))
    model.add(tf.keras.layers.LeakyReLU(0.2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    return model

