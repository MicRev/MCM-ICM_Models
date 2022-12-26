import numpy as np
from sko.ACA import ACA_TSP


def calTotalDistance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def getMatrix(points: np.array):
    """假设一图为一系列点的全连接构成, 由这些点坐标得到该图的邻接矩阵

    Parameters
    ----------
    points : np.array
        点坐标矩阵

    Returns
    -------
    np.array
        图的邻接矩阵
    """
    mat = []
    for i in range(points.shape[0]):
        mat.append([])
        for j in range(points.shape[0]):
            dist = np.linalg.norm(points[i] - points[j])
            mat[i].append(dist)
    return np.array(mat)


if __name__ == '__main__':
    distance_matrix = getMatrix(
        np.array([[0, 0],
                  [0, 1],
                  [2, 3],
                  [3, 0]])
    )
    aca = ACA_TSP(func=calTotalDistance, n_dim=4, size_pop=50, max_iter=100, distance_matrix=distance_matrix)
    bestX, bestY = aca.run()
    print(bestX)
