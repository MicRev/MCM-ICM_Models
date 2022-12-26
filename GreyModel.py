import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


class Model:
    def __init__(self, x, y, coef, b, residual):
        self.x = x
        self.y_hat = y
        self.coef = coef
        self.b = b
        self.residual = residual


def GM(x0: np.array, averageCorrect=False, order=1):
    wronskian = [x0]
    for i in range(order):
        x = [wronskian[i-1][0]]
        for j, xi in enumerate(wronskian[i][1:]):
            x.append(x[j] + xi)
        if averageCorrect:
            z = [x[0]]
            for j, xi in enumerate(x[1:]):
                z.append(0.5 * z[j] + 0.5 * xi)
            x = z
        wronskian.append(np.array(x))
    wronskian = np.array(wronskian)
    reg = linear_model.LinearRegression()
    reg.fit(wronskian[1:].T, wronskian[0].T)
    an = -reg.intercept_
    coef = -reg.coef_

    s = '微分方程为\nx^({})'.format(0)
    for n in range(order):
        c = ' + {}x^({})'.format(coef[n], n+1)
        s += c
    s += ' + {} = 0\n其中, x^(n)表示x的n阶和数列'.format(an)

    x_hat = np.matmul(np.array(coef), wronskian[1:])
    epsilon = sum((x_hat - x0) ** 2)
    s += '\n残差和: {}'.format(epsilon)

    res = Model(wronskian[1:], x_hat, -coef, -an, epsilon)
    print(s)
    return res


if __name__ == '__main__':
    x = np.array([1, 3, 9, 20, 50, 90, 100, 97, 65, 30, 14, 8, 4])
    epsilon = []
    for order in range(1, 21):
        res = GM(x, averageCorrect=True, order=order)
        epsilon.append(res.residual)
    plt.plot(range(1, 21), epsilon)
    plt.show()