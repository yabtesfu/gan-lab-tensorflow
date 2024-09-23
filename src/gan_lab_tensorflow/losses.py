from __future__ import annotations


def _tf():
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for GAN losses") from exc
    return tf


def vanilla_discriminator_loss(real_logits, fake_logits):
    tf = _tf()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_logits), real_logits)
    fake_loss = cross_entropy(tf.zeros_like(fake_logits), fake_logits)
    return real_loss + fake_loss


def vanilla_generator_loss(fake_logits):
    tf = _tf()
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_logits), fake_logits)


def wasserstein_discriminator_loss(real_logits, fake_logits):
    tf = _tf()
    return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)


def wasserstein_generator_loss(fake_logits):
    tf = _tf()
    return -tf.reduce_mean(fake_logits)


def gradient_penalty(discriminator, real_samples, fake_samples):
    tf = _tf()
    batch = tf.shape(real_samples)[0]
    alpha_shape = [batch] + [1] * (len(real_samples.shape) - 1)
    alpha = tf.random.uniform(alpha_shape, 0.0, 1.0)
    interpolated = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = tape.gradient(pred, interpolated)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=list(range(1, len(grads.shape)))) + 1e-12)
    return tf.reduce_mean(tf.square(slopes - 1.0))

