import numpy as np

class AHPModel:
    
    def __init__(self):
        self.weight = None  # 权重向量, n * 1, n为指标数
        self.judge = None  # 评分矩阵, m * n, m为评价目标数，n为指标数
    
    def result(self):
        return np.matmul(self.judge, self.weight)
    
    def _calConsisMat(self, mat: np.array, method: str) -> float:
        """从一致矩阵计算对应的权重

        Parameters
        ----------
        mat : np.array
            一致矩阵
        method : str
            计算方法。m: 算数平均值(mean); g: 几何平均值(geometric mean); e: 特征值(eigenvalue)

        Returns
        -------
        float
            计算出的权重

        Raises
        ------
        Exception
            若矩阵未通过一致性检验, 则报错.
        Exception
            若输入的方法不为m, g或e, 则认定方法未非法方法并报错.
        """
        if not self.isConsist(mat):
            raise Exception("矩阵未通过一致性检验")
        if method not in ["m", "g", "e"]:
            raise Exception("非法方法")
        res = np.zeros((mat.shape[0],))
        match method:
            case "m":  
                '''
                res = \vec{\omega}~
                \omega_i = \frac{1}{n} \sum_{j=1}^{n} \frac{a_{ij}}{\sum_{k=1}^{n} a_{kj}}
                '''
                for col in mat.T:
                    res += col / sum(col)
                res /= mat.shape[0]  # 归一化
            case "g":
                '''
                res = \vec{\omega}~
                \omega_i = \frac{\left(\prod_{j=1}^{n} a_{ij}\right)^{1/n} }{\sum_{k=1}^n \left(\prod_{j=1}^{n} a_{kj} \right)^{1/n}}
                '''
                def mul(l: iter):
                    res = 1
                    for i in l:
                        res *= i
                    return res
                
                for idx, row in enumerate(mat):
                    res[idx] += mul(row)
                res = res ** (1 / mat.shape[0])
                res /= sum(res)  # 归一化
            case "e":
                '''
                res 为归一化的最大特征值对应的特征向量
                '''
                eigs, eigvs = np.linalg.eig(mat)
                print(eigs)
                print(eigvs)
                res = eigvs[:, np.argmax(eigs)]
                res = np.real(res)
                res /= sum(res)
        return res
    
    def getJudge(self, judgeMats: list[np.array], method: str) -> np.array:
        """从不同指标的一致矩阵列表得到判断矩阵

        Parameters
        ----------
        judgeMats : list[np.array]
            一致矩阵列表
        method : str
            计算方法。m: 算数平均值(mean); g: 几何平均值(geometric mean); e: 特征值(eigenvalue)

        Returns
        -------
        np.array
            判断矩阵
        """
        res = np.zeros((judgeMats[0].shape[0], len(judgeMats)))
        res = res.T
        for idx, mat in enumerate(judgeMats):
            res[idx] = self._calConsisMat(mat, method)
        res = res.T
        self.judge = res
        return res
    
    def getWeight(self, weightMat: np.array, method: str) -> np.array:
        """从权重的一致矩阵得到权重向量

        Parameters
        ----------
        weightMat : np.array
            权重的一致矩阵
        method : str
            计算方法。m: 算数平均值(mean); g: 几何平均值(geometric mean); e: 特征值(eigenvalue)

        Returns
        -------
        np.array
            权重向量
        """
        res = self._calConsisMat(weightMat, method)
        self.weight = res
        return res
    
    def isConsist(self, mat: np.array) -> bool:
        RIs = [-1, 0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
        n = mat.shape[0]
        myRI = RIs[n]
        lmds, _ = np.linalg.eig(mat)
        myEI = (max(lmds) - n) / (n - 1)
        return myEI / myRI < 0.1    
    