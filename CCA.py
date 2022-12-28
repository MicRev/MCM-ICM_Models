import math
import numpy as np
        
def CCA(x_dataset: np.array, y_dataset: np.array) -> list[tuple[float, np.array, np.array]]:
    """由自变量样本矩阵x和因变量样本矩阵y得到若干对典型相关变量

    Parameters
    ----------
    x_dataset : np.array
        自变量样本矩阵, 大小为n * t, n为自变量维数, t为样本数量
    y_dataset : np.array
        因变量样本矩阵, 大小为m * t, m为因变量维数, t为样本数量

    Returns
    -------
    list[tuple[float, np.array, np.array]]
        每一对典型相关变量的列表, 其中每一对典型相关变量用三个参数表示: 相关系数, 自变量系数, 因变量系数
    """
    n, t = x_dataset.shape
    m, _ = y_dataset.shape
    A = np.zeros((n+m, t))

    A[:n, :] = x_dataset
    A[n:, :] = y_dataset

    for idx, array in enumerate(A):
        avg = np.mean(array)
        std = np.std(array)
        A[idx] = (array - avg) / std

    Cov = np.cov(A, bias = True)

    R_11 = np.matrix(Cov[:n, :n])
    R_12 = np.matrix(Cov[:n, n:])
    R_21 = np.matrix(Cov[n:, :n])
    R_22 = np.matrix(Cov[n:, n:])

    M = np.linalg.inv(R_11) * R_12 * np.linalg.inv(R_22) * R_21
    N = np.linalg.inv(R_22) * R_21 * np.linalg.inv(R_11) * R_12

    eig1, vector1 = np.linalg.eig(M)

    data = []

    for i in range(len(eig1)):
        if math.isclose(eig1[i], 0, abs_tol=1e-10):
            continue
        rho = np.sqrt(eig1[i])
        alpha = vector1[:, i]
        k = 1 / (alpha.T * R_11 * alpha)
        alpha *= np.sqrt(k)
        beta = np.linalg.inv(R_22) * R_21 * alpha / rho

        data.append((rho, alpha, beta))
    
    data.sort(key = lambda x: x[0], reverse = True)

    return data      
