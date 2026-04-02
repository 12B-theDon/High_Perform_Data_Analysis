import matplotlib.pyplot as plt
import time
import dis
import copy

#@profile
def save_figure(u, step):
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.title(f"Step {step}")
    plt.savefig(f"diffusion_step_{step}.png")
    plt.close()

#@profile : performane counter should be run without @profile
def diffusion(u_in, u_out, dt, D=1.0):
    xmax, ymax = data_size
    for i in range(xmax):
        for j in range(ymax):
            dxx = (u_in[(i + 1) % xmax][j] + u_in[(i - 1) % xmax][j] - 2.0 * u_in[i][j])
            dyy = (u_in[i][(j + 1) % ymax] + u_in[i][(j - 1) % ymax] - 2.0 * u_in[i][j])
            u_out[i][j] = u_in[i][j] + D * dt * (dxx + dyy) # u_out replaces u_new
    
def dropInk(max_iter):
    xmax, ymax = data_size
    u = [[0.0] * ymax for x in range(xmax)]
    u_new = [[0.0] * ymax for x in range(xmax)] # add u_new

    # initialization
    ink_low = int(data_size[0] * 0.4)
    ink_high = int(data_size[0] * 0.6)

    for i in range(ink_low, ink_high):
        for j in range(ink_low, ink_high):
            u[i][j] = 0.005
    
    u_init = copy.deepcopy(u)
    start = time.time()
    for i in range(max_iter):
        diffusion(u, u_new, 0.1)
        u, u_new = u_new, u
        
        #if i in save_steps:
        #    save_figure(u, i)

    end = time.time()

    return end-start, u_init, u

if __name__ == "__main__":
    data_size = ((640, 640))
    
    elapsed, u_init, u = dropInk(100)

#    dis.dis(save_figure)
#    dis.dis(diffusion)
#    dis.dis(dropInk)
    
    #plt.imshow(u)
    #plt.show()


# kernprof needs @profile, but perf doesnt need @profile
# kernprof -lv diffusion_2D_memoryReduction.py