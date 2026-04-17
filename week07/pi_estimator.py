import time
import numpy as np

def estimate_num_points_in_circle(num_samples):
    np.random.seed()
    xs = np.random.uniform(0, 1, num_samples)
    ys = np.random.uniform(0, 1, num_samples)
    quarter_unit_circle =(xs * xs + ys * ys) <= 1
    num_trials_in_quarter_unit_circle = np.sum(quarter_unit_circle)
    return num_trials_in_quarter_unit_circle


if __name__ == "__main__":
    num_samples_in_total = (1e8)
    print("Allocate {} samples per worker".format(num_samples_in_total))
    num_in_circle = 0
    start_time = time.time()
    num_in_circle += estimate_num_points_in_circle(int(num_samples_in_total))
    print("{} secs".format(time.time() - start_time))
    pi_estiamte = float(num_in_circle)/num_samples_in_total * 4
    print("Estimated PI", pi_estiamte)
    print("PI", np.pi)