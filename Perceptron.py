import numpy as np
import random


def step(x):
    if x > 0:
        return 1
    else:
        return 0


class Neuron:
    def __init__(self, weight=None):
        self.w = weight
        self.success = False
        self.count = 0

    def train(self, x_data: np.array, y_data: np.array, tolerance=0.01, rate=0.1, max_iter=50):
        # add [-1, -1, ..., -1] as the last column
        d = x_data.shape[1] + 1
        datas = x_data.shape[0]
        x_data = x_data.T
        np.append(x_data, np.array([-1] * datas))
        x_data.reshape(d, datas)
        x_data = x_data.T
        # randomly set self.w if necessary
        if not self.w:
            self.w = np.array([random.random() for _ in range(d)])

        y_hat = step(np.matmul(x_data, self.w))
        while sum((y_hat - y_data) ** 2) / datas > tolerance and self.count < max_iter:
            self.w += rate * (y_data - y_hat) * x_data
        return self.w

    def predict(self, x: np.array):
        return np.matmul(x, self.w)


if __name__ == '__main__':
    n = Neuron()

