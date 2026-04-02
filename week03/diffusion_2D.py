import matplotlib.pyplot as plt
import time
import dis

def save_figure(u, step):
    plt.figure()
    plt.imshow(u)
    plt.colorbar()
    plt.title(f"Step {step}")
    plt.savefig(f"diffusion_step_{step}.png")
    plt.close()

#@profile
def diffusion(u_in, dt, D=1.0):
    xmax, ymax = data_size
    u_new = [[0.0] * ymax for x in range(xmax)]
    for i in range(xmax):
        for j in range(ymax):
            dxx = (u_in[(i + 1) % xmax][j] + u_in[(i - 1) % xmax][j] - 2.0 * u_in[i][j])
            dyy = (u_in[i][(j + 1) % ymax] + u_in[i][(j - 1) % ymax] - 2.0 * u_in[i][j])
            u_new[i][j] = u_in[i][j] + D * dt * (dxx + dyy)
    return u_new

def dropInk(max_iter):
    xmax, ymax = data_size
    u = [[0.0] * ymax for x in range(xmax)]

    # initialization
    ink_low = int(data_size[0] * 0.4)
    ink_high = int(data_size[0] * 0.6)

    for i in range(ink_low, ink_high):
        for j in range(ink_low, ink_high):
            u[i][j] = 0.005
    
    u_init = u

    save_steps = {100, 200, 400, 800}  # O(1) lookup

    start = time.time()
    for i in range(max_iter):
        u = diffusion(u, 0.1)

        #if i in save_steps:
        #    save_figure(u, i)

    end = time.time()

    return end-start, u_init, u

if __name__ == "__main__":
    data_size = ((640, 640))
    
    elapsed, u_init, u = dropInk(100)
    print("time elapsed: ", elapsed)
    #dis.dis(save_figure)
    #dis.dis(diffusion)
    #dis.dis(dropInk)

    #plt.imshow(u)
    #plt.show()


# kernprof -lv diffusion_2D.py