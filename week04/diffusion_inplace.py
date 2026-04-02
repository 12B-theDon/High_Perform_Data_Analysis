import time
import numpy as np
import matplotlib.pyplot as plt


data_size = (640,640)

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
    
    #saved = {}

    start = time.time()
    
    for i in range(max_iter):
        diffusion(u, u_new, 0.1)
        u, u_new = u_new, u
        
        #if i in save_steps:
        #    saved[i] = u.copy()
    
    end = time.time()
    return end-start, u, #saved

# main loop for profiling
if __name__ == "__main__":
    save_steps = {100, 200, 400}

    elapsed, u = dropInk(1000)
    print("time elapsed: ", elapsed)

    #fig, axes = plt.subplots(1, len(save_steps), figsize=(15, 5))

    #for ax, step in zip(axes, sorted(save_steps)):
    #    im = ax.imshow(saved[step])
    #    ax.set_title(f"Step {step}")
    #    ax.axis("off")

    #fig.colorbar(im, ax=axes)
    #plt.tight_layout()
    #plt.show()

# sudo -i
# cd /home/mhlee/Documents/High_Performance_Data_Analysis/week04
# perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses,task-clock,faults,minor-faults,cs,migrations python3 diffusion_inplace.py

# kernprof -lv diffusion_inplace.py