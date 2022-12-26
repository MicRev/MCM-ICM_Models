# y = c^T x
# A_ub @ x <= b_ub
# A_eq @ x == b_eq

from scipy.optimize import linprog

if __name__ == '__main__':
    c = [-1, -2]
    A_ub = [[2, 1],
            [-4, 5],
            [1, -2]]
    b_ub = [20, 10, 2]
    A_eq = [[-1, 5]]
    b_eq = [15]

    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    print(res)
    print(res.x)

# .con 是等式约束残差。
# .fun 是最优的目标函数值（如果找到）。
# .message 是解决方案的状态。
# .nit 是完成计算所需的迭代次数。
# .slack 是松弛变量的值，或约束左右两侧的值之间的差异。
# .status是一个介于0和之间的整数4，表示解决方案的状态，例如0找到最佳解决方案的时间。
# .success是一个布尔值，显示是否已找到最佳解决方案。
# .x 是一个保存决策变量最优值的 NumPy 数组。
