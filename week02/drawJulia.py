# pip3 install matplotlib
import matplotlib.pyplot as plt

# Julia iteration 계산
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

# 좌표 생성 + Julia 계산
def build_Julia_set(desired_width, max_iterations):
    x1 = -1.8
    x2 = 1.8
    y1 = -1.8
    y2 = 1.8

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

    return output, len(x), len(y), x1, x2, y1, y2


# 이미지 저장
def save_julia_image(width=1000, iterations=30000000):
    output, w, h, x1, x2, y1, y2 = build_Julia_set(width, iterations)

    # 1D 리스트 → 2D 이미지
    image = []
    for i in range(h):
        image.append(output[i * w:(i + 1) * w])

    plt.figure(figsize=(8, 8))
    plt.imshow(image, extent=[x1, x2, y1, y2], cmap="inferno")
    plt.axis("off")

    plt.savefig("julia.png", dpi=300, bbox_inches="tight")
    print("Saved julia.png")


if __name__ == "__main__":
    save_julia_image()