import numpy as np
import typing
from overload import overload

class Topsis:

    def __init__(self, mat: np.array) -> None:
        self.mat = mat

    @overload
    def posit(self, cols: list[int]) -> np.array:
        """将矩阵正向化, 针对极小型指标

        Parameters
        ----------
        cols : list[int]
            指定指标的列号列表

        Returns
        -------
        np.array
            正向化后的矩阵
        """
        mat = self.mat.T
        for idx in cols:
            mat[idx] = max(mat[idx]) - mat[idx]
        self.mat = mat.T
        return mat.T
    
    @overload
    def posit(self, cols: list[int], x_best) -> np.array:
        """"将矩阵正向化, 针对极小型指标

        Parameters
        ----------
        cols : list[int]
            指定指标的列号列表
        x_best : float
            预期的中间值

        Returns
        -------
        np.array
            正向化后的矩阵
        """
        mat = self.mat.T
        for idx in cols:
            M = max([abs(i - x_best) for i in mat[idx]])
            newCol = np.array([1-abs(xi - x_best) / M for xi in mat[idx]])
            mat[idx] = newCol
        self.mat = mat.T
        return mat.T
    
    @overload
    def posit(self, cols: list[int], x_min, x_max) -> np.array:
        """将矩阵正向化, 针对极小型指标

        Parameters
        ----------
        cols : list[int]
            指定指标的列号列表
        x_min : float
            区间最小值
        x_max : float
            区间最大值

        Returns
        -------
        np.array
            正向化后的矩阵
        """
        mat = self.mat.T
        for idx in cols:
            M = max(x_min - min(mat[idx]), max(mat[idx] - x_max))
            array = np.array([1 if x_min <= xi <= x_max else 1 - (x_min - xi) / M if xi < x_min else 1 - (xi - x_max) / M for xi in mat[idx]])
            '''
            if xi < x_min:
                xi = 1 - (x_min - xi) / M
            elif x_min <= x <= x_max:
                xi = 1
            elif x > x_max:
                xi = 1 - (x_i - x_max) / M
            '''
        self.mat = mat.T
        return mat.T
    
    def _normalize(self):
        mat = self.mat
        mat = mat.T
        for idx, col in enumerate(mat):
            mat[idx] = col / np.sqrt(sum([xi**2 for xi in col]))
        self.mat = mat.T
        return mat.T
    
    def score(self) -> np.array:
        """根据评价矩阵计算得分

        Returns
        -------
        np.array
            得分向量
        """
        self._normalize()
        mat = self.mat
        maxes = [max(col) for col in mat.T]
        mins = [min(col) for col in mat.T]
        res = np.zeros((mat.shape[0],))
        for idx, row in enumerate(mat):
            dmax = np.sqrt(sum([(maxes[j]-row[j])**2 for j in range(mat.shape[1])]))
            dmin = np.sqrt(sum([(mins[j]-row[j])**2 for j in range(mat.shape[1])]))
            res[idx] += dmin / (dmax + dmin)
        res /= sum(res)
        return res