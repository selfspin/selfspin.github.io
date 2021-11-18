---
title: '贝叶斯最优分类器'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---



从贝叶斯最优分类器推导常见的线性分类器，包括朴素贝叶斯，线性判别分析和Fisher判别分析。

<!--more-->

## Naive Bayes

朴素贝叶斯分类器几乎是最简单的分类器，但在很多应用场景却极为强大，背后来自于Bayes最优准则的作用。

从 [概率图](https://truenobility303.github.io/PGM/) 的角度来看，朴素贝叶斯假设在给定因变量$Y$的条件下，特征$X_1,...,X_p$ 是条件独立的。

狭义的朴素贝叶斯模型针对于离散型特征$X$，多用于文本分类等任务, 此时的特征为每个词的示性函数或者词频数。



### Classification

Bayes最优准则简单来讲就是选择给定条件下最大的概率的一类，是对于最小化错误率的最优分类器，原因是错误率相当于使用0-1损失为损失函数，


$$
\min E[I(y \ne f(x) \vert x)] = \min P(y \ne f(x) \vert x) = \max P(y = f(x) \vert x) 
$$


从简单的推导可以看出，选择后验概率最大的类别，可以使得分类器达到错分率的理论下界，从这个角度来说Bayes分类器是最优的。

由于我们假设特征$X$是由类别$Y$所产生的，因此需要使用Bayes公式计算上述的条件概率，又由于


$$
\max_Y P( Y \vert X) = \max_Y \frac{P(X,Y)}{P(X)} = \max_Y \frac{P(X \vert Y) P(Y)}{\sum_Y P(X \vert Y) P(Y)}
$$


对于分母上面的$P(X)$，只需利用$Y$进行求和即可得到。

再根据条件独立性假设，$p$个特征共现的概率只需要简单乘积就好了，


$$
\max_Y P(Y \vert X_1,...,X_p) = \max_Y \frac{\prod_{j=1}^p P(X_j \vert Y) P(Y)}{\sum_Y \prod_{j=1}^p  P(X \vert Y) P(Y)}
$$


### Parameter Estimation

朴素贝叶斯中的参数估计非常简单，只需要用频率估计概率就好了。

但其实朴素贝叶斯模型的背后是存在着统计模型的假设的，用频率估计概率是多项分布的极大似然估计的性质，因此朴素贝叶斯模型假设了类别$Y$的先验分布和$P(X \vert Y)$ 都服从多项分布，下面简单可以推导出多项分布的极大似然估计，


$$
P(Y =k) = \pi_1^{m1} \pi_2^{m_2}... \pi_K^{m_K} , m_k = \delta_k
$$


写出似然函数的表达式，


$$
\begin{align}
\log \mathcal{L} &= \log \prod_{i=1}^N P(Y_i) = \sum_{i=1}^N \sum_{k=1}^K m_{ik} \log \pi_k  = \sum_{k=1}^K n_k \log \pi_k , \text{With } n_k = \sum_{i=1}^N  m_{ik} \\
\end{align}
$$


考虑到概率的归一化约束，引入Largange乘子并且求导，


$$
\begin{align}
L &=  \sum_{k=1}^K n_k \log \pi_k - \lambda (\sum_{k=1}^K \pi_k-1) \\
\frac{dL}{d \pi_k} &= \frac{n_k}{\pi_k} - \lambda = 0  
\end{align}
$$


利用到约束条件可以求解得到极大似然估计的结果，


$$
\begin{align}
n_k &= \lambda \pi_k \\
\sum_{k=1}^K n_k &=  \lambda  \sum_{k=1}^K \pi_k = \lambda  \\
\pi_k &= \frac{n_k}{\lambda} = \frac{n_k}{\sum_{k=1}^K n_k} 
\end{align}
$$


利用Lagrange乘子系数$\lambda$正好充当了归一化因子的角色，极大似然估计的结果正好是根据频率估计概率。



## Linear Discrimination Analysis

### Classification

线性判别分析可以看作是朴素贝叶斯模型在特征$X$为连续场景下的推广，其假设贝叶斯分类器中的条件分布为同方差的高斯分布，




$$
p(x \vert y) = \frac{1}{\sqrt{2 \pi \det \Sigma}} \exp(-\frac{1}{2}(x - \mu_k) ^T \Sigma^{-1} (x- \mu_k))
$$


而对于标签$y$仍然假设其满足参数为$\pi_k$的多项分布，计算后验概率的比例以便于选取最大的后验概率进行Bayes分类，


$$
\begin{align}
\log \frac{p(y = \pi_k \vert x)}{p(y = \pi_0 \vert x)} &= \log \frac{p(x \vert y= \pi_k) p(y= \pi_k)}{p(x \vert y= \pi_0) p(y= \pi_0)}\\
&=  (x^T \Sigma^{-1} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log \pi_k) - (x^T \Sigma^{-1} \mu_0 - \frac{1}{2}\mu_0^T \Sigma^{-1} \mu_0 + \log \pi_0 )\\
&= x^T \Sigma^{-1} (\mu_k-\mu_0) - \frac{1}{2}(\mu_k + \mu_0)^T \Sigma^{-1} (\mu_k-\mu_0) + \log \frac{\pi_k}{\pi_0}\\
&= w^T x +b  \\
\text{With } w &=  \Sigma^{-1} (\mu_k-\mu_0), b = - \frac{1}{2}(\mu_k + \mu_0)^T \Sigma^{-1} (\mu_k-\mu_0) + \log \frac{\pi_k}{\pi_0}
\end{align}
$$


对于二分类问题，可以直接计算上述比例得到决策边界，而对于多分类问题, 计算下面的线性函数取最大值即可，


$$
\max_k x^T \Sigma^{-1} \mu_k - \frac{1}{2}\mu_k^T \Sigma^{-1} \mu_k + \log \pi_k
$$


### Parameter Estimation

使用极大似然估计，在该模型中，极大似然估计都是好求解的。

对于参数$\pi_k$，采用多项分布的极大似然估计，也即采用频率估计概率

对于每一个高斯分布的均值和方差，根据熟知的结论，也只要使用样本均值和样本方差估计即可，下面利用矩阵代数简要证明该结论，利用求导的方式最大化对数似然函数：


$$
\begin{align}
\mathcal{L} &= \log \prod_{i=1}^N \frac{1}{\sqrt{2 \pi \det \Sigma}} \exp(-\frac{1}{2} (x_i - \mu)^T \Sigma^{-1} (x_i - \mu)) \\ 
&= -\frac{N}{2} \log \det \Sigma - \frac{1}{2} \sum_{i=1}^N (x_i - \mu)^T \Sigma^{-1} (x_i -\mu) \\
&= -\frac{N}{2} \log \det \Sigma - \frac{1}{2} \sum_{i=1}^N tr(\Sigma^{-1} (x_i -\mu) (x_i - \mu)^T )\\
\end{align}
$$


首先求解均值的极大似然估计，得到的结果就将是样本均值，


$$
\begin{align}
&\frac{d \mathcal{L}}{d \mu} = -\sum_{i=1}^N \Sigma^{-1} (x_i - \mu)\\
&\sum_{i=1}^N \Sigma^{-1} (x_i - \mu) =  0 \\
& \sum_{i=1}^N (x_i - \mu) =  0 \\
& \hat \mu = \bar x
\end{align}
$$


利用矩阵代数的公式，
$$
\begin{align}
d tr(AX) &= tr(A dX) \\
d \log \det X &= tr(X^{-1} dX) \\
dX^{-1} &= -X^{-1} dX X^{-1}
\end{align}
$$


继续求解方差的极大似然估计，并且嵌入对于均值的估计结果，


$$
\begin{align}
d \mathcal{L} &= -\frac{N}{2} tr(\Sigma^{-1} d \Sigma ) -\frac{1}{2} \sum_{i=1}^N tr(d \Sigma^{-1} (x_i -\mu) (x_i - \mu)^T ) \\
&= -\frac{N}{2} tr(\Sigma^{-1} d \Sigma ) - \frac{1}{2}  \sum_{i=1}^Ntr(-\Sigma^{-1} d\Sigma \Sigma^{-1}(x_i -\mu) (x_i - \mu)^T ) \\
\frac{d \mathcal{L}}{d \Sigma} &= -\frac{N}{2} \Sigma^{-1} + \frac{1}{2} \sum_{i=1}^N \Sigma^{-1}(x_i -\mu) (x_i - \mu)^T  \Sigma^{-1} = 0\\
\end{align}
$$


可以得到方差的极大似然估计就是样本方差，


$$
\begin{align}
-\frac{N}{2} \Sigma^{-1} + \frac{1}{2}  \Sigma^{-1} \sum_{i=1}^N(x_i -\hat \mu) (x_i - \hat \mu)^T  \Sigma^{-1} &= 0 \\
-\frac{1}{2} \Sigma^{-1} + \frac{1}{2}  \Sigma^{-1} S \Sigma^{-1} &= 0,\text{With } S= \frac{1}{N}\sum_{i=1}^N(x_i -\hat \mu) (x_i - \hat \mu)^T \\
S \Sigma^{-1} &= I \\
\Sigma &= S
\end{align}
$$


总结在线性判别分析模型中，在训练过程等价于求解参数的估计，


$$
\begin{align}
\pi_k &= \frac{n_k}{N} \\
\mu_k & = \bar X_k \\
\Sigma_k &= S_k
\end{align}
$$
上式中的$n_k,\bar X_k,S_k$ 分别表示训练集中属于第$k$类的个数，样本均值，样本方差.

---

有趣的是，如果将标签作为隐变量，此时将是一个无监督的聚类任务，此时导出的模型是 [高斯混合模型](https://truenobility303.github.io/GMM-PLSA/) 

在此我们可以看到有监督学习和无监督学习之间千丝万缕的联系，而朴素贝叶斯算法，也可以对应于假设分布为多项分布的 [主题模型](https://truenobility303.github.io/GMM-PLSA/) 或者称为概率潜在语义分析。

## Fisher  Discrimination Analysis

Fisher判别分析的思想来自于对投影后的方差进行刻画，类似于 [PCA](https://truenobility303.github.io/PCA/) 的思想。

与作为无监督学习的降维算法PCA不同的是，Fisher判别分析是一个有监督的降维方法。



### Multi-Class Classification

PCA中的优化目标可以理解为最大化方差，而Fisher判别分析中由于带有标签信息，需要先给出组间方差$S_w$和组内方差$S_b$的定义, 在上述定义的前提下，可以推出总方差可以被分解为，



$$
\begin{align}
S_t &= \sum_{i=1}^N (X_i - \mu) (X_i - \mu)^T \\
&= \sum_{i=1}^N \sum_{f(X_i) = Y_j} (X_i - \mu_j + \mu_j - \mu )(X_i - \mu_j + \mu_j - \mu )^T \\
&= \sum_{i=1}^N \sum_{f(X_i) = Y_j}(X_i - \mu_j)(X_i - \mu_j)^T+ \sum_{i=1}^N \sum_{f(X_i) = Y_j}(\mu_j - \mu)(\mu_j - \mu)^T \\
&= S_w +S_b
\end{align}
$$



等式成立的原因是交换求和顺序后可以发现倒数第二步的交叉项为零。

上式也可以写成矩阵形式，并且可以计算得到进行投影$\tilde X = XW$后的方差,


$$
\begin{align}
S_t  &= X^T H X,\text{With } H = I - \frac{1}{N}ee^T  \\
\tilde S_t &= W^T X^T H XW = W^T S_t W
\end{align}
$$


类似地可以得到投影后的组间方差和组内方差都应该满足，


$$
\begin{align}
\tilde S_t &=  W^T S_t W \\
\tilde S_w &=  W^T S_w W  \\
\tilde S_b &=  W^T S_b W  \\
\end{align}
$$


Fisher判别分析希望找到一个合适的投影矩阵$W$，使得组间方差最大，此时不同类别之间的区分度更大，同时也应该使得组内方差尽可能小，此时同类别的数据会尽量被投影在一个位置。如果找到了满足上述条件的$W$矩阵之后，对于新的数据的分类任务，只需要利用投影矩阵$W$投影，并且观察投影后$X$距离哪个类别的均值$\mu$被投影之后的距离更近即可。



利用矩阵的迹，将上述问题转化为一个优化问题，并且使用基于梯度的方法求解，


$$
\begin{align}
\min L & =\min \frac{1}{2}tr (W^TS_b W)^{-1} (W^T S_w W)  \\
dL &=  tr((W^TS_b W)^{-1} S_w WdW  - (W^TS_b W)^{-1} S_bW dW (W^TS_b W)^{-1} (W^T S_w W))   \\
&=tr([W^TS_b W)^{-1} S_wW -(W^TS_b W)^{-1} (W^T S_w W)(W^TS_b W)^{-1} S_bW]dW) \\
\frac{dL}{dW} &= (W^TS_b W)^{-1} S_wW -(W^TS_b W)^{-1} (W^T S_w W)(W^TS_b W)^{-1} S_bW = 0\\
\end{align}
$$



对于驻点的等式进行化简可以得到，


$$
\begin{align}
(W^TS_b W)^{-1} S_wW &= (W^TS_b W)^{-1} (W^T S_w W)(W^TS_b W)^{-1} S_bW \\
S_w W&= (W^T S_w W)(W^TS_b W)^{-1} S_bW \\
\end{align}
$$


为了求解上面的式子，使用半正定矩阵的同时合同对角化操作，下面先简要推导该技巧，


$$
\begin{align}
W^T S_w W & = LL^T , \text{By Cholesky Decomposition} \\
L^TW^T S_b W L&= U \Lambda U^T , \text{By Spectral Decomposition} \\
P^T  W^TS_wW P &=(LU)^T W^TS_wW(LU) = U^TU = I ,\text{Let } P = LU\\
P^T  W^TS_bW P &=(LU)^T W^TS_bW(LU) = \Lambda \\ 
\end{align}
$$


利用同时合同对角化之后的结果可以对等式进行化简，


$$
\begin{align}
S_w W &= (W^T S_w W)(W^TS_b W)^{-1} S_bW  \\
S_w W &= (PP^T) (P \Lambda P^T)^{-1} S_b W \\
S_w W &= P \Lambda^{-1} P^{-1} S_b W \\
\Lambda P^{-1} S_w W &= P^{-1} S_b W 
\end{align}
$$


最终的问题转化为求解矩阵$(P^{-1} S_w, P^{-1}S_b)$的广义特征值问题，其对应的特征向量构成了矩阵$W$

---

如果考虑一种更为简单的优化目标，结果将更为简单，


$$
\begin{align}
\min \frac{tr(W^TS_w  W)}{tr(W^T S_b W)} = \min tr(W^TS_w  W), \text{ s.t. }  tr(W^T S_b W) = 1
\end{align}
$$


利用Lagrange乘子法求解上述问题，


$$
\begin{align}
L &=  \frac{1}{2} [tr(W^T S_w W) - \lambda (tr(W^T S_b W)-1)] \\
\frac{dL}{dW} &= S_w W - \lambda S_b W = 0
\end{align}
$$


得到的解也是一个广义特征值问题，


$$
\begin{align}
S_w W &= \lambda S_b W \\
\frac{tr(W^TS_w  W)}{tr(W^T S_b W)} &= \frac{tr(\lambda W^TS_b  W)}{tr(W^T S_b W)} = \lambda
\end{align}
$$


因此，采用该优化目标得到的矩阵$W$是矩阵对$(S_w,S_b)$最小的广义特征值所对应的特征向量所构成的矩阵。



### Bayes Optimal Property

Fisher判别分析出发点是基于方差的角度，和基于贝叶斯分类器出发的线性判别分析出发点截然不同，但一个重要的性质是，如果假设类别的先验$\pi_k$服从均匀分布，且每一类的方差相同的二分类任务，Fisher判别分析和线性判别分析是等价的。



回归线性判别分析的判别准则，令判别超平面为0可以得到决策边界，


$$
\begin{align}
x^T \Sigma^{-1} (\mu_1-\mu_0) - \frac{1}{2}(\mu_1 + \mu_0)^T \Sigma^{-1} (\mu_1-\mu_0) + \log \frac{\pi_1}{\pi_0} &= 0 \\
x^T \Sigma^{-1} (\mu_1-\mu_0) - \frac{1}{2}(\mu_1 + \mu_0)^T \Sigma^{-1} (\mu_1-\mu_0)  &= 0,\text{With } \pi_1 = \pi_0 \\
\end{align}
$$


而对于二分类任务中Fisher判别分析，首先我们关注于投影方向的选择，由于Fisher判别分析中仅由方向决定，投影向量前的常数并无太大所谓，据此可以得到，


$$
\begin{align}
S_w w &= \lambda S_b w \\
S_w w &= \lambda (\mu_1 - \mu_0)(\mu_1- \mu_0)^T w \\
S_w w & = \lambda a (\mu_1 - \mu_0) ,\text{Let } a = (\mu_1- \mu_0)^T w\\
w & = \lambda a S_w^{-1} (\mu_1 - \mu_0) \\
w &= S_w^{-1} (\mu_1 - \mu_0), \text{By Ignoring the Constant Item } \lambda a \\
w &= \Sigma^{-1} (\mu_1 - \mu_0)
\end{align}
$$


在决策边界，应该有距离两类数据点的均值在$w$投影相等，并且正好为相反数，


$$
\begin{align}
(w^T x - w^T \mu_0) + (w^T x - w^T \mu_1) &=0 \\
2x^T \Sigma^{-1} (\mu_1 - \mu_0) - (\mu_1 + \mu_0)^T \Sigma^{-1} (\mu_1 - \mu_0) &= 0
\end{align}
$$


对比上下两个式子可以发现正好得到了完全相同的决策边界，


$$
x^T \Sigma^{-1} (\mu_1-\mu_0) - \frac{1}{2}(\mu_1 + \mu_0)^T \Sigma^{-1} (\mu_1-\mu_0)  = 0
$$


因此，两个出发点不同的分类器在该位置殊途同归，共同达到了贝叶斯最优分类器的目标。