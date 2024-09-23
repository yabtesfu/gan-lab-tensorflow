# Experiment Plan

1. Start with the quadratic distribution from the tutorial notes.
2. Track real and generated samples every few hundred steps.
3. Add sine and mixture distributions to test mode collapse.
4. Compare vanilla GAN loss with WGAN-GP.
5. Add replay-buffer experiments to see whether stale generated samples stabilize the discriminator.
6. Add conditional labels for class-aware mixture generation.
7. Track MMD and nearest-neighbor precision alongside visual samples.
8. Enable light discriminator augmentation when real accuracy stays too high.
9. Add MNIST or Fashion-MNIST once the synthetic loop is stable.
10. Record generator/discriminator loss curves and sample snapshots.
