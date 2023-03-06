# 简介

本文档为不同模型代码实现的接口介绍, 以及使用时须注意的事项.

* [层次分析法](#层次分析法)
* [蚁群算法](#蚁群算法)
* [典型相关分析](#典型相关分析)
* [图结构](#图结构)
* [灰色预测模型](#灰色预测模型)
* [插值法](#插值法)
* [线性规划](#线性规划)
* [整数线性规划](#整数线性规划)
* [最短路径](#最短路径)
* [最小生成树](#最小生成树)
* [函数的多项式拟合](#函数的多项式拟合)
* [相关性](#相关性)
* [Topsis 优劣解距离法](#topsis-优劣解距离法)

# 层次分析法

提供对权重和各评价指标的判断矩阵, 计算权重向量和各评价目标在各指标下的评价矩阵, 并得到评价结果.
由AHPModel类实现.

[AHP.py](AHP.py)

## AHPModel类

实现矩阵储存和计算的对象.

## 内部变量

`weight`：权重向量，$n \times 1$，$n$为指标数
`judge`：评分矩阵，$m \times n$，$m$为评价目标数，$n$为指标数

## 类函数

### `getJudge()`

从不同指标的一致矩阵列表得到判断矩阵

**Args**:

    judgeMats (list[np.array]): 一致矩阵列表
    method (str): 计算方法。m: 算数平均值(mean); g: 几何平均值(geometric mean); e: 特征值(eigenvalue)

**Returns**:

    np.array: 判断矩阵

### `getWeight()`

从权重的一致矩阵得到权重向量

**Args**:

    weightMat (np.array): 权重的一致矩阵
    method (str): 计算方法。m: 算数平均值(mean); g: 几何平均值(geometric mean); e: 特征值(eigenvalue)

**Returns**:

    np.array: 权重向量

### `isConsist()`

判断一个矩阵是否为一致矩阵

**Args**:

    mat (np.array): 待判断的矩阵

**Returns**:

    (bool): 是否为一致矩阵

### `result()`

得到评价结果

**Returns**:

    (np.array): 评价结果向量

## 使用例

```python
from AHP import AHPModel
import numpy as np

ahpModel = AHPModel()  # 创建类对象
weightMat = ...  # 权重的一致矩阵
ahpModel.getWeight(weightMat, "g")  # 通过几何平均值求权重矩阵
arg1Mat = ...
arg2Mat = ...
...
argMats = [arg1Mat, arg2Mat, ...]  # 不同评价指标的一致矩阵
ahpModel.getJudge(argMats, "e")  # 通过特征值法求得判断矩阵
print(ahpModel.result())  # 得到各评价目标的评分

# 注意: 若输入的矩阵不是一致矩阵, 将报错. 若不确定是否为一致矩阵，请用isConsist方法判断.
```

# 蚁群算法

启发式算法, 求解遍历图所有节点一次并回到起点, 使得连接这些节点的路径成本最低.
由sko.ACA中的`ACA_TSP()`函数实现, [Ants.py](Ants.py)中提供了更便捷的函数接口.

## 内置函数

### `calTotalDistance()`

由路径计算距离, 提供给`ACA_TSP()`的第一个参数

### `getMatrix()`

假设一图为一系列点的全连接构成, 由这些点坐标得到该图的邻接矩阵

**Args**
    points (np.array): 点坐标矩阵

**Returns**
    np.array: 图的邻接矩阵

## 使用例

```python
import numpy as np
from sko.ACA import ACA_TSP
from Ants import calTotalDistance, getMatrix

distance_matrix = getMatrix(
    np.array([[0, 0],
              [0, 1],
              [2, 3],
              [3, 0]])
)
aca = ACA_TSP(func=calTotalDistance, n_dim=4, size_pop=50, max_iter=100, distance_matrix=distance_matrix)
bestX, bestY = aca.run()
print(bestX)
```

# 典型相关分析

分析多维自变量$x$和多维因变量$y$之间的相互作用, 通过寻找若干对典型变量$U$, $V$, 使得

$$
U_i = \sum_{j} a_{ij}x_j,~
V_i = \sum_{j} b_{ij}y_j
$$

之间有最大的相关系数, 且不同组的典型相关变量之间相关系数为0.

[CCA.py](CCA.py)

## 内置函数

### CCA()

由自变量样本矩阵x和因变量样本矩阵y得到若干对典型相关变量

**Args**:

`x_dataset : np.array` 自变量样本矩阵, 大小为n * t, n为自变量维数, t为样本数量
`y_dataset : np.array` 因变量样本矩阵, 大小为m * t, m为因变量维数, t为样本数量

**Returns**:

`list[tuple[float, np.array, np.array]]` 每一对典型相关变量的列表, 其中每一对典型相关变量用三个参数表示: 相关系数, 自变量系数, 因变量系数

## 使用例

```python
from CCA import CCA
import numpy as np

x = np.array(...)
y = np.array(...)  # 数据矩阵

res = CCA(x, y)
print(res[0])  # 相关系数最大的典型相关系数
```

# 图结构

实现图的数据结构, 实现其与邻接矩阵的相互转换, 并为其他基于图的算法提供接口

[graph.py](graph.py)

## Node类

图节点结构的实现

### 内部变量

`index : int`: 节点的序号, 节点携带信息和区分不同节点的主要依据
`neighbours : list[node]`: 邻接的节点
`weight: list[Any]`: 邻接节点对应的边的权重
`previous : Node`: 仅在[MinPath.py](MinPath.py)中用到, 生成最短路径后该节点的前一节点

### 类函数

#### `addNeighbour()`

 将另一个节点以给定权重与该节点相连, 节点加入`Node.nodes`, 权重加入`Node.weights`

**Args**:

    node (Node): 待连接的另一节点
    weight (int): 两节点间边的权重

#### `getWeight()`

获取与另一节点间边的权重. 若这两个节点未相连, 返回`-1`.

**Args**:

    node (Node): 期望得到相连边权重的另一节点

**Returns**:

    int: 相连的边的权重, 若未相连则为-1

## Graph 类

图结构的实现. 

### 内部变量

`size : int`: 图包含的节点数, 即图的大小
`nodes : list[Node]`: 图包含的节点列表, 按 `index` 从 `0` 到 `size-1` 排序

### 类函数

#### `generate()`

由邻接矩阵生成图结构

**Args**:

    adj_matrix (np.array): 邻接矩阵

#### `getAdjMatrix()`

得到该图的邻接矩阵

**Returns**:

    np.array: 图的邻接矩阵

#### `clear()`

将所有节点的`previous`置为`None`. 由于`previous`属性仅在部分模块中使用, 且在某几个函数中使用后就不再被需要, 因此需要使用`clear`函数将`previous`置`None`, 触发垃圾回收机制并节省内存.


# 灰色预测模型

通过指定阶数的微分方程拟合一系列离散点, 返回微分方程的表达式. 这类微分方程都有如下形式：

$$
c_0 x+c_1 \frac{d}{dt}x + c_2 \frac{d^2}{dt^2}x+\cdots+c_n \frac{d^n}{dt^n}x = 0
$$

[GreyModel.py](GreyModel.py)

## 内置函数

### `GM()`

通过任意维数的离散数据建立微分方程预测，返回对应的预测值和模型准确度信息. 该函数会在终端输出模型的微分方程形式和残差和等信息.

**Args**:

    x0 (np.array): 待拟合的离散点
    averageCorrect (bool): 是否采用平均值修正计算高阶加和数据, 缺省为False
    order (int): 采用的微分方程的阶数, 缺省为1

**Returns**:

    Model: 预测结果的相关信息, 见下

## Model类

储存输出结果的类, 包含结果所需的各类信息.

### 内部变量

`x : np.array` 离散点的高阶加和项
`y_hat : np.array` 对原离散数据由该方程生成的预测值
`coef : np.array` 高阶加和项的系数向量
`b : np.array | float` 拟合的截距
`residual : float` 模型残差和

## 使用例

```python
from GreyModel import GM
import numpy as np

x = np.array([...])
res = GM(x, order=2)  # 使用二阶微分方程拟合
```

# 插值法

由一系列已知点的坐标拟合可能的函数, 并给出给出横坐标的未知点的纵坐标. 
由spline(三次样条插值)和pchip(Hermit插值)两种函数实现

[interpolation.py](interpolation.py)

## 内置函数

### `spline()`

三次样条插值

**Args**:

    xs (np.array): 已知点的横坐标
    ys (np.array): 已知点的纵坐标
    new_x (float): 插值点的横坐标

**Returns**:

    float: 插值点的纵坐标

### `pchip()`

Hermit插值
**注意该函数只能用于递增的序列**

**Args**:

    xs (np.array): 已知点的横坐标
    ys (np.array): 已知点的纵坐标
    new_x (float): 插值点的横坐标

**Returns**:

    float: 插值点的纵坐标

## 使用例

```python
import numpy as np
from interpolation import spline, pchip

xs = np.array([...])
ys = np.array([...])  # 已知点的坐标
newx = 2.45
# newx = [2.45, 1.36, 4.78]
newy = spline(xs, ys, newx)  # 三次样条插值获得插值点的纵坐标
print(newy)
```

# 线性规划

使用scipy.optimize包中的linprog函数求解线性规划问题.

## 使用例

```python
from scipy.optimize import linprog

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
```

# 整数线性规划

使用分支定界法求解变量要求为整数的线性规划问题。

[IntLinProg.py](IntLinProg.py)

## 内置函数

### `BranchBoard()`

当一个整数线性规划问题被如下描述:

最优化函数

`y = c^T @ x`
    
当x满足以下条件:
    
`A_ub @ x <= b_ub`
`A_eq @ x == b_eq`
    
且x中各项为整数.

其中, A_eq 和 b_eq 为可选参数. 此类问题即可用该分支定界函数求解.

**Args**:

`c : np.array` 最优化函数中x的系数向量
`A_ub : np.array` 约束条件中各不等式的系数矩阵
`b_ub : np.array` 约束条件中各不等式的结果向量
`A_eq : np.array, optional` 约束条件中各等式的系数矩阵, 缺省为None
`b_eq : np.array, optional` 约束条件中各等式的结果向量, 缺省为None

**Returns**:

`float` 最优化函数的值
`np.array` 最优化结果的x值

**Raises**:

`Exception` 若非整数线性规划问题无解, 则整数线性规问题必定也无解, 此时抛出异常"没有合理解"

## 使用例

```python
import numpy as np
from IntLinProg import BranchBoard

c = np.array([-3, -2])
A_ub = np.array([[2, 3],
                 [2, 1]])
b_ub = np.array([14, 9])
ans = BranchBoard(c, A_ub, b_ub)
print(ans)
```

# 最短路径

使用Dijkstra算法求解从某一点出发遍历一图的最短路径问题.

[MinPath.py](MinPath.py)

## 内置函数

### `dijkstra()`

由图中某一个点开始找到图中每个节点的最短路径. 函数会在终端中输出最短路径的形式和距离, 并返回一个仅有最短路连接的图。

**Args**:

`graph : Graph` 图对象
`start : int` 起始点的序号

**Returns**:

`Graph` 新的图对象, 在原有的图的基础上只保留最短径而删除其他所有连接.

## 使用例

```python
import numpy as np
from MinPath import dijkstra()
from graph import Graph

graph = Graph(6)
A = np.array(...)  # 图的邻接矩阵
graph.generate(A)
shortest = dijkstra(graph, 0)
print(shortest.getAdjMatrix())
```

# 最小生成树

由一张图生成其最小生成树, 使用prim算法.

[MinSpawnTree.py](MinSpawnTree.py)

## 内置函数

### `prim()`

由图结构得到其最小生成树. 函数会返回作为最小生成树的图对象, 并在终端中输出最小生成树的边权和.

**Args**:

`graph : Graph` 图对象
`root : int, optional` 生成树的根节点, 缺省值为-1, 若缺省则随机选择一点作为根节点

**Returns**:

`Graph` 图对象, 仅保留作为最小生成树的边

## 使用例

```python
import numpy as np
from MinSpawnTree import prim
from graph import Graph

graph = Graph(6)
A = np.array(...)
graph.generate(A)
new_graph = prim(graph)
print(new_graph.getAdjMatrix())
```

# 函数的多项式拟合

使用numpy包的`polyfit()`函数和scipy包的`curve_fit()`函数进行多项式函数或任意参数函数的最小二乘拟合.

## 使用例

```python
import numpy as np
from scipy.optimize import curve_fit

x = ...
y = ...  # 待拟合的数据

f1 = np.polyfit(x, y, 3)  # 3为拟合多项式的最高次, 返回系数列表
p1 = np.poly1d(f1)  # p1为拟合多项式的表达式

f = lambda x, a, b, c: a * x + b ** x + c
pOpt, pCov = curve_fit(f, x, y)  # pOpt为最小二乘系数, pCov为各系数的协方差, 对角线即为各系数的方差
```

# 相关性

使用`scipy.stats`包中的`pearsonr()`函数和`spearmanr()`函数分别计算两列数据的pearson相关系数和spearman相关系数。

## 使用例

```python

import scipy.stats

x = [...]
y = [...]  # 待检验的两列数据

print(scipy.stats.pearsonr(x, y)[0])  # pearson相关系数
print(scipy.stats.spearmanr(x, y)[0])  # spearman相关系数

```

# Topsis 优劣解距离法

提供一个无数据缺失的评分矩阵, 由topsis方法得到各评价对象的评分.
由Topsis类实现.

[topsis.py](topsis.py)

## Topsis类

实现矩阵储存和计算的类对象.

## 内部变量

`mat`: 需处理的评价矩阵

## 类函数

### `posit()`

将矩阵正向化, 针对极小型指标.
该函数已重载, 依据不同变量数量对不同类型的指标(极小型、中间型、区间型)进行正向化

**Args**:

    cols (list[int]): 极小型指标的列号列表

**Args**:

    cols (list[int]): 中间型指标的列号列表
    x_best: 预期的中间值

**Args**:

    cols (list[int]): 区间型指标的列号列表
    x_min: 区间最小值
    x_max: 区间最大值

**Returns**:

    np.array: 正向化后的矩阵

### `score()`

根据评价矩阵计算得分, 得到的结果是归一化的.

**Returns**:

    np.array: 得分向量

## 使用例

```python
import numpy as np
from topsis import Topsis

t = Topsis  # 创建类对象
t.mat = np.array(...
                 ...)  # 输入原始数据矩阵
t.posit([0, 2, ...])  # 极小型指标的正向化
t.posit([1, 3, ...], x_best=10)  # 中间型指标的正向化
t.posit([4, 5, ...], x_min=2, x_max=7)  # 区间型指标的正向化
print(t.score())  # 计算评价结果  
``` 
