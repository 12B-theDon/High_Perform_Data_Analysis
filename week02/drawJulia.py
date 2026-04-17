import os

import matplotlib.pyplot as plt

from julia1 import build_Julia_set


def build_julia_image(width=1000, iterations=300):
    output = build_Julia_set(width, iterations)

    # `build_Julia_set` returns a flat list for a square grid.
    image = [output[i * width:(i + 1) * width] for i in range(width)]
    extent = [-1.8, 1.8, -1.8, 1.8]
    return image, extent


def save_julia_image(width=1000, iterations=300, output_path="results/julia.png"):
    image, extent = build_julia_image(width, iterations)

    plt.figure(figsize=(8, 8))
    plt.imshow(image, extent=extent, cmap="inferno")
    plt.axis("off")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    save_julia_image()
