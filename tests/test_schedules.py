from gan_lab_tensorflow.schedules import TTURSchedule, discriminator_steps_for_phase


def test_ttur_schedule_warms_up():
    schedule = TTURSchedule(generator_lr=0.1, discriminator_lr=0.2, warmup_steps=10)
    assert schedule.value_at(5) == (0.05, 0.1)


def test_discriminator_phase_adds_warmup_step():
    assert discriminator_steps_for_phase(10, warmup_until=100, base_steps=1) == 2
    assert discriminator_steps_for_phase(100, warmup_until=100, base_steps=1) == 1

