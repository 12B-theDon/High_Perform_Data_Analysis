N = 500
water = [0] * N
ink = [1] * 100

u = 0
D = 1
t = 0
dt = 0.1

u_init = water
u_init[200: 300] = ink
u = u_init

while True:
    u_new = [0] * N
    for i in range(N):
        u_new[i] = u[i] + D * dt * (u[(i + 1) % N] + u[(i - 1) % N] - 2 * u[i])
    u = u_new
    t += 1

    if t==500:
        break