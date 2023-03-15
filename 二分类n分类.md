## Logistic回归

### 适用场景

假设得出了对一组变量 $\boldsymbol{x}$ 的线性回归 $z = \boldsymbol{w}^{T}\boldsymbol{x}$，我们试图用 $y \in \{0, 1\} $ 来描述 $z$ 的二分类行为, 我们相信存在 $Sigmoid$函数可以这样估计y为1的概率：
$$
P(y=1|\boldsymbol{x}) = \sigma(\boldsymbol{w}^{T}\boldsymbol{x}) = \frac{e^{\boldsymbol{w}^{T}\boldsymbol{x}}}{1+e^{\boldsymbol{w}^{T}\boldsymbol{x}}}
$$
则有
$$
P(y=0|\boldsymbol{x}) = \sigma(\boldsymbol{w}^{T}\boldsymbol{x}) = \frac{1}{1+e^{\boldsymbol{w}^{T}\boldsymbol{x}}}
$$

我们有如此信仰的原因是，如果把一个事件的对数几率定义为
$$
logit(p) = \ln \frac{p}{1-p}
$$

那么 $y=1$ 的对数几率为
$$
logit(P(y=1|x)) = \ln \frac{P(y=1|x)}{1-P(y=1|x)} = \boldsymbol{w}^{T}\boldsymbol{x}
$$

也就是说，$y=1$ 的对数几率是 $\boldsymbol{x}$ 的线性函数，则 $y=1$ 的几率是 $z=\boldsymbol{w}^{T}\boldsymbol{x}$ 的 $Sigmoid$ 函数

$$
P(y=1|x) = \sigma(z) = \frac{1}{1+e^{-\boldsymbol{w}^{T}\boldsymbol{x}}}
$$
（上文和此处的 $Sigmoid$ 函数是相同的）

很容易注意到，$Sigmoid$ 函数与 $logit$ 函数互为反函数，所以之所以选择对数概率作为评判标准，是因为 $Sigmoid$ 函数的性质足够好。$Sigmoid$ 函数是一个将 $(-\infty, \infty)$ 映射到 $(0, 1)$ 之间的函数，且在0左侧和右侧迅速趋近于0和1，适合作为二分类的判别函数。

### 实现细节

使用最大似然法求解 $Sigmoid$ 函数的相关系数。

我们想用一个表示概率的函数 $f(y_i, \boldsymbol{x}_i, \boldsymbol{\omega})$ 来表示系统处于当前状态 $(y_i, \boldsymbol{x}_i, \boldsymbol{\omega})$ 的概率。一般认为其是 $(y_i, \boldsymbol{x}_i)$ 的函数，但极大似然法认为在 $(y_i, \boldsymbol{x}_i)$ 已知的情况下，参数 $\boldsymbol{\omega}$ 应该选择能使概率最大的值。

定义极大似然函数

$$
L(\boldsymbol{\omega} | y, \boldsymbol{x}) = \prod_i f(y_i, \boldsymbol{x}_i, \boldsymbol{\omega})
$$

在Logistic回归中，有

$$
f(y_i, \boldsymbol{x}_i, \boldsymbol{\omega}) =
\left\{
\begin{aligned}
    \sigma(\boldsymbol{\omega} \boldsymbol{x}_i )&, y_i = 1 \\
    1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i )&, y_i=0
\end{aligned}
\right.
$$

或

$$
f(y_i, \boldsymbol{x}_i, \boldsymbol{\omega}) =
(\sigma(\boldsymbol{\omega} \boldsymbol{x}_i ))^{y_i}(1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i))^{1-y_i}
$$

则

$$
\ln L = \sum \ln f= \sum (y_i \ln\sigma(\boldsymbol{\omega} \boldsymbol{x}_i ) + (1-y_i) \ln(1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i)))
$$

在给定 $y$ 和 $\boldsymbol{x}$ 的情况下，找到使 $\ln L$ 最大的 $\boldsymbol{\omega}$ 值，可使用梯度下降法。

不难证明：
$$
\frac{d}{dx} \sigma(x) = \sigma(x)(1-\sigma(x))
$$

那么
$$
\frac{d}{dx}\ln \sigma(x) = \frac{1}{\sigma(x)} \frac{d}{dx} \sigma(x) = 1-\sigma(x)
$$

因此对极大似然函数的对数值求导：
$$
\begin{aligned}
    \frac{\partial \ln L}{\partial \omega_j} & =\frac{\partial }{\partial \omega_j} \sum_i (y_i \ln\sigma(\boldsymbol{\omega} \boldsymbol{x}_i ) + (1-y_i) \ln(1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i)))\\
    & =  \sum_i \frac{\partial }{\partial \boldsymbol{\omega} \boldsymbol{x}_i} (y_i \ln\sigma(\boldsymbol{\omega} \boldsymbol{x}_i ) + (1-y_i) \ln(1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i))) \cdot \frac{\partial \boldsymbol{\omega} \boldsymbol{x}_i}{\partial \omega_j}\\
    & = \sum_i (y_i(1-\sigma(\boldsymbol{\omega} \boldsymbol{x}_i)) - (1-y_i)\sigma(\boldsymbol{\omega} \boldsymbol{x}_i)) \cdot x_{ij} \\
    & = \sum_i (y_i - \sigma(\boldsymbol{\omega} \boldsymbol{x}_i)) \cdot x_{ij}
\end{aligned}
$$

在该方向上梯度下降

训练时，需要区分训练集和测试集，如果可能，使用交叉验证（Cross Validation）

### 所需假设

### 可能风险

过拟合

## 支持向量机

### 适用场景

求解凸二次优化的最优化算法。
线性或非线性（使用核技巧）二分类模型

基本思想是找到能够正确划分数据集且支持向量距离最小的超平面。假设数据集为 $\{(\boldsymbol{x}_1, y_1), (\boldsymbol{x}_2, y_2), \cdots, (\boldsymbol{x}_n, y_n) \} $，则要找到超平面 $\boldsymbol{\omega}\boldsymbol{x} + b = 0$

某一个点到该超平面的几何距离为
$$
d_i = \frac{1}{||\boldsymbol{\omega}||} y_i (\boldsymbol{\omega}\boldsymbol{x}_i + b)
$$

支持向量距离超平面最近的数据点。则要找到的超平面的支持向量到超平面的距离为
$$
d = \min d_i
$$

### 实现细节

由以上定义，我们可以给出
$$
d_i = \frac{1}{||\boldsymbol{\omega}||} y_i (\boldsymbol{\omega}\boldsymbol{x}_i + b) \ge d
$$

即，支持向量机需要在如上条件下完成对 $d$ 的最优化 $\max d$

推导繁琐，直接调包。如需更多，阅读[文稿](https://zhuanlan.zhihu.com/p/31886934)

若几乎线性可分（去除几个点后线性可分），可规定错误样本的惩罚程度，并加上误差项来计算。

对于非线性分类而言，可以通过非线性变换将它转化为某个维特征空间中的线性分类问题，在高维特征空间中学习线性支持向量机。

由于在线性支持向量机学习的对偶问题里，目标函数和分类决策函数都只涉及实例和实例之间的内积，所以不需要显式地指定非线性变换，而是用核函数替换当中的内积。

核函数是一个二元函数 $K(x,z)$, 对任意 $x$ 特征空间的映射 $\phi(x)$，核函数计算了两个特征空间向量的内积：
$$
K(x, z) = <\phi(x), \phi(y)>
$$

核函数的引入一方面减少了我们计算量，另一方面也减少了我们存储数据的内存使用量。

常见核函数：
- 线性核
- 多项式核
- 高斯核

### 所需假设

- 原问题是凸优化问题
- 原问题满足[kkt条件](https://zhuanlan.zhihu.com/p/556832103)
- 任务是小批量任务，对时间和空间性能要求不太高
  
### 可能的风险

由于涉及到大批量矩阵运算，算力可能受限的情况下计算时间可能较长。

## Fischer线性判别

### 适用场景

或线性判别分析（Linear Discriminant Analysis，LDA）

给定训练集，讲样例投影到一维直线上，使同样类别的点尽可能靠近，不同类别的点尽可能远离。

也就是说，LDA对给定类别的数据进行有监督的降维，方法是将输入数据投影到一个线性子空间，该空间由使类与类之间的分离最大化的方向组成。

### 实现细节

本质上，LDA是基于贝叶斯概率估计的。假设一个数据点 $\boldsymbol{x}$ 的分类为 $y$，则该数据点的分类为某一给定分类 $y=k$ 的概率是：
$$
P(y=k|\boldsymbol{x}) = \frac{P(\boldsymbol{x}|y=k)P(y=k)}{P(\boldsymbol{x})} = \frac{P(\boldsymbol{x}|y=k)P(y=k)}{\sum_i P(\boldsymbol{x}|y=i)P(y=i)}
$$

并且，我们认为数据点在直线上的投影符合多元高斯分布

$$
P(\boldsymbol{x}|y=k) = \frac{1}{(2\pi)^{(d/2)} \left|\Sigma_k \right|^{1/2}} \exp(-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{\mu}_k)^T \Sigma_k^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_k))
$$

其中，$d$ 是特征的数目，$\Sigma_k$ 是协方差矩阵，$\mu_k$ 是样品均值，$(\boldsymbol{x}-\boldsymbol{\mu}_k)^T \Sigma_k^{-1}(\boldsymbol{x}-\boldsymbol{\mu}_k)$ 被定义为样品 $\boldsymbol{x}$ 与均值 $\boldsymbol{\mu}$ 之间的**马氏距离**（Mahalanobis Distance）

要预测该数据点具体属于哪一类，可以找到 $k$ 使得 $P(y=k|\boldsymbol{x})$ 最大。

关于如何选取一维直线的数学推导，可以参考[知乎](https://zhuanlan.zhihu.com/p/51769969)。

### 所需假设

- 数据线性可分
- 数据在给出的一维直线上符合一元高斯分布（中心极限定理）

### 可能的风险

- 过拟合
- 数据并不线性可分，此时可考虑QDA（二次判别分析）

## 决策树

### 适用场景

决策树是一种树形结构，每一个内部节点代表一个属性上的判断，每个分支代表一个分类结果。也就是说，在给定一系列已知分类结果的数据的情况下，通过学习这些数据，得到一棵决策树，通过这个决策树可以对新的数据进行判断分类。

决策树的优点在于极强的解释性，可以通过节点的分类标准得到人类可以轻易理解的判断依据。

### 实现细节

一般而言，决策树的生成主要由经以下两步：
1. 节点的分裂
   当一个节点的属性无法给出判断，将一个节点分裂为2个节点。
2. 阈值的确定
   选择恰当的阈值使分类错误最小。

比较常用的决策树模型有ID3，C4.5，CART（Classification And Regression Tree）

决策树学习中可能遇到过拟合问题，可用剪枝方法进行处理，分为预剪枝和后剪枝。

预剪枝是指在决策树生成**过程中**，对每个结点在划分前进行估计，若当前节点的划分不能带来决策树泛化能力提升，则停止划分并将当前节点标记为叶节点。

后剪枝则是先从训练集生成一颗**完整的决策树**，然后自底向上地对非叶节点进行考察，若将该节点对应的子树替换为叶节点能带来决策树泛化能力提升，则将该节点替换为叶节点。

#### ID3

由[信息熵](https://baike.baidu.com/item/%E4%BF%A1%E6%81%AF%E7%86%B5/7302318)决定哪个节点作为父节点，哪个节点需要分裂开。由信息熵的熵增原理，对一组数据来说，信息熵越小说明分类结果越好。

信息熵的定义为

$$
S = -\sum_i [p(x_i)\cdot \log_2 p(x_i)]
$$

$p(x_i)$ 为情况 $x_i$ 出现的概率

#### C4.5

ID3在分类上有一定问题，它认为越小的分割分类错误率越小。因此，ID3可能导致过细的分割，从而导致过拟合。

因此，在C4.5模型中，优化项会除以分割太细的代价，将这个比值称为信息增益率。

#### CART

CART是一棵二叉树，也是回归树，同时也是分类树。

CART只将一个父节点分为两个子节点，它用GINI指数决定如何分类。

GINI指数类似于熵的概念，总体内包含的类别越杂乱，GINI指数就越大。对数据集D，

$$
Gini(D) = \sum_{k=1} \sum_{k' \ne k} p_k p_{k'} = 1-\sum_k p_k^2
$$

即，从数据集内随机抽两个样本，其类别不一样的概率。



### 所需假设


### 可能风险
