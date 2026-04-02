import time
import numpy as np
import matplotlib.pyplot as plt

def save_figure(u):
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.savefig(__file__.rsplit(".", 1)[0] + ".png")
    plt.close()

#@profile
def diffusion_op(u_in):
    return (
        np.roll(u_in, +1, axis=0)
        + np.roll(u_in, -1, axis=0)
        + np.roll(u_in, +1, axis=1)
        + np.roll(u_in, -1, axis=1)
        - 4 * u_in
    )

#@profile
def diffusion(u_in, dt, D=1.0):
    return u_in + dt * D * diffusion_op(u_in)

#@profile
def dropInk(max_iter, save_steps=None):
    u = np.zeros(data_size, dtype=float)

    ink_low = int(data_size[0] * 0.4)
    ink_high = int(data_size[0] * 0.6)
    u[ink_low:ink_high, ink_low:ink_high] = 0.005

    if save_steps is None:
        save_steps = set()

    start = time.time()

    for i in range(max_iter):
        u = diffusion(u, 0.1)

    end = time.time()

    return end - start, u

if __name__ == "__main__":
    data_size = (640, 640)

    save_steps = {100, 200, 400}

    elapsed, u = dropInk(500, save_steps=save_steps)
    save_figure(u)
    print("time elapsed:", elapsed)

# time elapsed: 1.7161164283752441
# kernprof -lv diffusion_numpy.py
# perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 diffusion_numpy.py