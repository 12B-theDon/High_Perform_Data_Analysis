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
def custom_roll(rollee, shift, axis, out):
    if shift==1 and axis==0:
        out[1:, :] += rollee[:-1, :]
        out[0, :] += rollee[-1, :]
    elif shift == -1 and axis == 0:
        out[:-1, :] += rollee[1:, :]
        out[-1, :] += rollee[0, :]
    elif shift == 1 and axis == 1:
        out[:, 1:] += rollee[:, :-1]
        out[:, 0] += rollee[:, -1]
    elif shift == -1 and axis == 1:
        out[:, :-1] += rollee[:, 1:]
        out[:, -1] += rollee[:, 0]

#@profile
def diffusion_op(u_in, u_out):
    np.copyto(u_out, u_in)
    u_out *= -4
    custom_roll(u_in, +1, 0, u_out)
    custom_roll(u_in, -1, 0, u_out)
    custom_roll(u_in, +1, 1, u_out)
    custom_roll(u_in, -1, 1, u_out)

#@profile
def diffusion(u_in, u_out, dt, D=1):
    diffusion_op(u_in, u_out)
    u_out *= D * dt
    u_out += u_in

#@profile
def dropInk(max_iter):
    u = np.zeros(data_size)
    u_new = np.zeros(data_size)
    ink_low = int(data_size[0] * 0.4)
    ink_high = int(data_size[0] * 0.6)
    u[ink_low:ink_high, ink_low:ink_high] = 0.005
    start = time.time()
    
    for i in range(max_iter):
        diffusion(u, u_new, 0.1)
        u, u_new = u_new, u
    end = time.time()
    return end-start, u

# main loop for profiling
if __name__ == "__main__":
    data_size = (640,640)
    elapsed, u = dropInk(1000)
    save_figure(u)
    print("time elapsed:", elapsed)

# time elapsed: 1.7757248878479004
# kernprof -lv diffusion_custom.py
# perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 diffusion_custom.py