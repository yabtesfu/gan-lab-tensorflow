from gan_lab_tensorflow.data import sample_curve, summarize


if __name__ == "__main__":
    points = sample_curve(512, seed=42)
    print(summarize(points))

