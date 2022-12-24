# 简介

本文档为不同模型代码实现的接口介绍, 以及使用时须注意的事项.

* [层次分析法](#层次分析法)
* [Topsis 优劣解距离法](#topsis-优劣解距离法)
* [插值法](#插值法)

# 层次分析法

提供对权重和各评价指标的判断矩阵, 计算权重向量和各评价目标在各指标下的评价矩阵, 并得到评价结果.
由AHPModel类实现.

[AHP.py](AHP.py)

## 类

`class AHPModel`

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

    from AHP import AHPModel

    ahpModel = AHPModel  # 创建类对象
    weightMat = ...  # 权重的一致矩阵
    ahpModel.getWeight(weightMat, "g")  # 通过几何平均值求权重矩阵
    arg1Mat = ...
    arg2Mat = ...
    ...
    argMats = [arg1Mat, arg2Mat, ...]  # 不同评价指标的一致矩阵
    ahpModel.getJudge(argMats, "e")  # 通过特征值法求得判断矩阵
    print(ahpModel.result())  # 得到各评价目标的评分

    # 注意: 若输入的矩阵不是一致矩阵, 将报错. 若不确定是否为一致矩阵，请用isConsist方法判断.

# Topsis 优劣解距离法

提供一个无数据缺失的评分矩阵, 由topsis方法得到各评价对象的评分.
由Topsis类实现.

[topsis.py](topsis.py)

## 类

`class Topsis`

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

    from topsis import Topsis

    t = Topsis  # 创建类对象
    t.mat = np.array(...
                     ...)  # 输入原始数据矩阵
    t.posit([0, 2, ...])  # 极小型指标的正向化
    t.posit([1, 3, ...], x_best=10)  # 中间型指标的正向化
    t.posit([4, 5, ...], x_min=2, x_max=7)  # 区间型指标的正向化
    print(t.score())  # 计算评价结果   

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

    from interpolation import spline, pchip

    xs = np.array([...])
    ys = np.array([...])  # 已知点的坐标
    newx = 2.45
    # newx = [2.45, 1.36, 4.78]
    newy = spline(xs, ys, newx)  # 三次样条插值获得插值点的纵坐标
    print(newy)
