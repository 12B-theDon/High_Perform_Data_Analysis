import time
from julia1 import calculate_Julia_set

def printVer_build_Julia_set(desired_width, max_iterations):
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

	# 여기서부터 변형
	print("Length of x: ", len(x))
	print("Total elements: ", len(zs))
	start_time = time.time()
	output = calculate_Julia_set(max_iterations, zs, cs)
	end_time = time.time()
	secs = end_time - start_time
	print(calculate_Julia_set.__name__ + " took", secs, "seconds to execute.")
	# 여기까지 변형

	return output