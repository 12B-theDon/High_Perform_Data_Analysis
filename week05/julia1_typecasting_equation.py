import time
import cythonfn_typecasting_equation

x1 = -1.8
x2 = 1.8
y1 = -1.8
y2 = 1.8

c_real = -0.8
c_imag = 0.156

def build_julia_set(desired_width, max_iterations):
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x = []
    y = []
    ycoord = y2

    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step

    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    zs = []
    cs = []

    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    start_time = time.time()
    output = cythonfn_typecasting_equation.calculate_julia_set(max_iterations, zs, cs)
    print(time.time() - start_time, " sec")

if __name__ == "__main__":
    build_julia_set(desired_width=2000, max_iterations=500)

# for typecasting equation
# 0.699089765548706  sec