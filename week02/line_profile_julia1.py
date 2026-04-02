@profile
def no_ABS_calculate_Julia_set(max_iter, zs, cs):
    output = [0] * len(zs)

    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]

        while True:

            if n >= max_iter:
                break

            zr = z.real
            zi = z.imag

            if zr * zr + zi * zi >= 4:
                break

            z = z * z + c
            n += 1

        output[i] = n

    return output

@profile
def calculate_Julia_set(max_iter, zs, cs):
    output = [0] * len(zs)
    for i in range(len(zs)):
      n = 0
      z = zs[i]
      c = cs[i]
      while abs(z) < 2 and n < max_iter:
        z = z * z + c
        n += 1
      output[i] = n
    return output

#---------------------------------------------------#
@profile
def build_Julia_set(desired_width, max_iterations):
    x1 = -1.8
    x2 = 1.8
    y1 = -1.8
    y2 = 1.8
    c_real = 4
    c_real = -0.67172
    c_imag = -0.42193

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

    output = calculate_Julia_set(max_iterations, zs, cs)
    #output = no_ABS_calculate_Julia_set(max_iterations, zs, cs)

    return output

if __name__ == "__main__":
    build_Julia_set(1000, 300)
    