import time
import numpy as np
import matplotlib.pyplot as plt

data_size = (640, 640)


def diffusion_op(u_in):
    return (
        np.roll(u_in, +1, axis=0)
        + np.roll(u_in, -1, axis=0)
        + np.roll(u_in, +1, axis=1)
        + np.roll(u_in, -1, axis=1)
        - 4 * u_in
    )


def diffusion(u_in, dt, D=1.0):
    return u_in + dt * D * diffusion_op(u_in)


def dropInk(max_iter, save_steps=None):
    u = np.zeros(data_size, dtype=float)

    ink_low = int(data_size[0] * 0.4)
    ink_high = int(data_size[0] * 0.6)
    u[ink_low:ink_high, ink_low:ink_high] = 0.005

    if save_steps is None:
        save_steps = set()

    saved = {}  # step별 결과 저장

    start = time.time()

    for i in range(max_iter):
        u = diffusion(u, 0.1)

        #if i in save_steps:
        #    saved[i] = u.copy()  # 중요: copy 안 하면 다 마지막 상태됨

    end = time.time()

    return end - start, u, #saved


if __name__ == "__main__":
    save_steps = {100, 200, 400}

    elapsed, u = dropInk(500, save_steps=save_steps)
    print("time elapsed:", elapsed)

    #fig, axes = plt.subplots(1, len(save_steps), figsize=(15, 5))

    #for ax, step in zip(axes, sorted(save_steps)):
    #    im = ax.imshow(saved[step])
    #    ax.set_title(f"Step {step}")
    #    ax.axis("off")

    #fig.colorbar(im, ax=axes)
    #plt.tight_layout()
    #plt.show()