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
def diffusion_op(u_in, u_out):
    np.copyto(u_out, u_in)
    u_out *= -4
    u_out += np.roll(u_in, +1, 0)
    u_out += np.roll(u_in, -1, 0)
    u_out += np.roll(u_in, +1, +1)
    u_out += np.roll(u_in, -1, +1)

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

    save_steps = {100, 200, 400}

    elapsed, u = dropInk(1000)
    save_figure(u)
    print("time elapsed:", elapsed)

# time elapsed: 1.9314076900482178
# kernprof -lv diffusion_inplace.py
# perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 diffusion_inplace.py