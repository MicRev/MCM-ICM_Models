# y = c^T x
# A_ub @ x <= b_ub
# A_eq @ x == b_eq

from scipy.optimize import linprog
import numpy as np
import sys

def _isInt(num: float):
    if np.isclose(num, int(num), 1e-6) or \
            np.isclose(num, int(num) + 1, 1e-6) or \
            np.isclose(num, int(num) - 1, 1e-6):
        return True
    else:
        return False


def BranchBoard(c: np.array, A_ub: np.array, b_ub: np.array, A_eq=None, b_eq=None) -> tuple[float, np.array]:
    """通过分支定界法求解整数线性规划问题. 该问题的详细描述为: 最优化函数:
    
    `y = c^T @ x`
    
    当x满足以下条件:
    
    `A_ub @ x <= b_ub`
    
    `A_eq @ x == b_eq`
    
    且x中各项为整数.

    Parameters
    ----------
    c : np.array
        最优化函数中x的系数向量
    A_ub : np.array
        约束条件中各不等式的系数矩阵
    b_ub : np.array
        约束条件中各不等式的结果向量
    A_eq : np.array, optional
        约束条件中各等式的系数矩阵, 缺省为None
    b_eq : np.array, optional
        约束条件中各等式的结果向量, 缺省为None

    Returns
    -------
    float
        最优化函数的值
    np.array
        最优化结果的x值

    Raises
    ------
    Exception
        若非整数线性规划问题无解, 则整数线性规划问题必定也无解, 此时抛出异常"没有合理解"
    """
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    if not res.success:  # if LP question has no certain solution, ILP has no solution too.
        raise Exception("没有合理解")
    if all(_isInt(x) for x in res.x):
        x = res.x
        val = np.dot(ans.x, c)
        return val, x
    for idx, xi in enumerate(res.x):
        if not _isInt(xi):
            break  # when idx is the index of the float
    A_add1 = np.zeros(A_ub.shape[1])
    A_add1[idx] = 1
    A_add2 = A_add1.copy()
    A_add2[idx] = -1
    shape = A_ub.shape
    # left branch (x_i <= [x_i])
    A_ub1 = np.append(A_ub, A_add1)
    b_ub1 = np.append(b_ub, np.floor(xi))
    A_ub1 = A_ub1.reshape((shape[0]+1, shape[1]))
    # right branch (x_i >= [x_i] + 1)
    A_ub2 = np.append(A_ub, A_add2)
    b_ub2 = np.append(b_ub, - np.ceil(xi))
    A_ub2 = A_ub2.reshape((shape[0]+1, shape[1]))

    val1, x1 = BranchBoard(c, A_ub1, b_ub1, A_eq, b_eq)  # DFS
    val2, x2 = BranchBoard(c, A_ub2, b_ub2, A_eq, b_eq)
    if val1 <= val2:
        return val1, x1  # get the miner one
    return val2, x2


if __name__ == '__main__':
    c = np.array([-3, -2])
    A_ub = np.array([[2, 3],
                     [2, 1]])
    b_ub = np.array([14, 9])
    ans = BranchBoard(c, A_ub, b_ub)
    print(ans)