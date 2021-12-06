---
title: '稀疏优化与低秩优化初步'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---



矩阵填充是推荐系统中的关键问题，本文以此问题的求解为例说明矩阵次梯度等一些稀疏优化、低秩优化等的基础内容，同时也包括鲁棒PCA的介绍。



<!--more-->

本文主要关注于矩阵上的优化解法，因此首先关于矩阵函数的梯度。

## Matrix Gradient

矩阵梯度用方向导数定义，


$$
\begin{align}
\nabla f(A) &= D \\
\text{Iff } \lim_{t \rightarrow 0} \frac{f(A+tB)- f(A)}{t} &= \langle B,D \rangle ,\forall B \\ 
\end{align}
$$


其实类似对于泛函的导数也具有类似的定义，可以参见 [指数族分布](https://truenobility303.github.io/EF/) 中对于其EM算法的变分法的推导部分，



下面给出几个常见的矩阵函数的梯度，



**Example1** $f(X) = tr(X^T A) = \langle X,A \rangle$

$$
\begin{align}
\lim_{t \rightarrow 0} \frac{f(X+tY)- f(X)}{t} &= \lim_{t \rightarrow 0} \frac{\langle tY,A \rangle}{t} = \langle Y,A \rangle  \\
\end{align}
$$

因此，$\nabla f(X) = A$



**Example2** $f(X) = tr(X^{-1})$
$$
\begin{align}
\lim_{t \rightarrow 0} \frac{f(X+tY)- f(X)}{t} &=  \lim_{t \rightarrow 0} \frac{tr(X+tY)^{-1}) - tr(X^{-1})}{t} \\
&=  \lim_{t \rightarrow 0} \frac{tr(I +t X^{-1}Y)^{-1}X^{-1}- tr(X^{-1})}{t} \\
&=  \lim_{t \rightarrow 0} \frac{tr(I-tX^{-1}Y +o(t^2))X^{-1}- tr(X^{-1})}{t} ,\text{With Taylor's Expansion} \\
&= \lim_{t \rightarrow 0} \frac{tr(I-tX^{-1}Y +o(t^2))X^{-1}- tr(X^{-1})}{t} \\
&= -tr(X^{-1} Y X^{-1}) \\
&=  -tr(X^{-2} Y) \\
&= -\langle X^{-2T}, Y \rangle \\
\end{align}
$$
因此，$\nabla f(X) = -X^{-2T}$



**Example3** $f(X) = \log \det X$


$$
\begin{align}
\lim_{t \rightarrow 0} \frac{f(X+tY)- f(X)}{t} &=  \lim_{t \rightarrow 0} \frac{\log \det X+tY - \log \det X}{t} \\
&=\lim_{t \rightarrow 0} \frac{\log \det X(I+tX^{-1}Y) - \log \det X}{t} \\
&=\lim_{t \rightarrow 0} \frac{\log \det tX^{-1}Y}{t} \\ 
&= \lim_{t \rightarrow 0} \frac{\sum_i t\lambda_i(X^{-1}Y)}{t} \\
&= \sum_i \lambda_i(X^{-1}Y) \\
&= tr(X^{-1}Y) \\
&=\langle X^{-T},Y \rangle
\end{align}
$$


因此，$\nabla f(X) = X^{-T}$



## Matrix Sub-Gradient

对于连续不可导的凸函数，应该推广梯度的定义，称为次梯度，本质上为凸函数的支撑，矩阵次梯度定义为满足下式的$G$构成的集合


$$
\begin{align}
G &\in \partial f(A) \\
\text{Iff } f(B) - f(A) &\ge \langle G, B-A \rangle  
\end{align}
$$


可以知道对于可导函数次梯度就是其梯度，但次梯度对于不可导函数将是一个集合。且可以证明对于一个凸函数，当$ 0 \in \partial A$的时候取到其极小值，这也可以看作可导凸函数的极值定理的推广。



由于常用矩阵范数作为正则化项，将上述次梯度的定义用于矩阵范数上可以得到，


$$
\begin{align}
G &\in \partial \Vert A \Vert \\
\text{Iff } \Vert B \Vert - \Vert A \Vert &\ge \langle G, B-A \rangle  
\end{align}
$$


例如在鼓励稀疏性的优化中，我们用$L_0$范数（非零元的个数）的最佳凸近似$L_1$范数（绝对值的和）来近似$L_0$范数。

而$L_1$范数的次梯度可以根据定义直接得到，根据每个元素的独立性，


$$
\begin{align}
(\partial \Vert X \Vert_1)_{ij} &=  [-1,1] ,\text{ If } X_{ij} = 0 \\
(\partial \Vert X \Vert_1)_{ij} &=  1 ,\text{ If } X_{ij} > 0 \\
(\partial \Vert X \Vert_1)_{ij} &=  -1 ,\text{ If } X_{ij} < 0 \\
\end{align}
$$


定义符号函数$sgn(x)$ 表示上面的逐元素的次梯度，可以得到矩阵形式的表达，


$$
\partial \Vert X \Vert_1 = sgn(X)
$$


### Sub-Gradient of Matrix Norm

并且对于矩阵范数这种特殊的函数，可以证明其次梯度$G$有如下的等价定理，其中 $\Vert G \Vert_{\star}$表示对偶范数，


$$
\begin{align}
G &\in \partial \Vert A \Vert \\
\text{Iff } \Vert A \Vert  &= \langle A,G \rangle \\
\text{And } \Vert G \Vert_{\star} &\le 1, \text{With } \Vert G \Vert_{\star} = \max_{\Vert B \Vert =1} \langle G,B\rangle
\end{align}
$$


首先证明该定理的必要性，取特殊的$B$代入次梯度的定义即可，


$$
\begin{align}
\langle A ,G \rangle &\ge \Vert A \Vert , \text{Let } B = O \\
\langle A ,G \rangle &\le \Vert A \Vert , \text{Let } B = 2A \\
\text{Then} \Vert A \Vert &= \langle A,G \rangle 
\end{align}
$$
再代入次梯度的定义中，得到


$$
\begin{align}
\Vert B \Vert - \Vert A \Vert &\ge \langle G, B-A \rangle   \\
\Vert B \Vert - \langle A,G \rangle  &\ge \langle G, B-A \rangle \\
\Vert B \Vert &\ge \langle G,B \rangle \\
\text{Then } \Vert G \Vert_{\star} &\le 1
\end{align}
$$


对其其充分性，证明是类似而显然地，


$$
\begin{align}
\Vert B \Vert - \Vert A \Vert &\ge \langle G, B-A \rangle \\
\Vert B \Vert &\ge \langle G,B \rangle ,\text{With } \Vert A \Vert = \langle G,A \rangle,\Vert G \Vert_{\star} \le1 \\
\end{align}
$$




### Sub-Gradient of Schatten p-norm

在正式求解问题之前，还需要引入Schatten-p范数的定义，其定义为奇异值的范数，且常用的几个矩阵范数都为Schatten-p范数。


$$
\begin{align}
\Vert A \Vert_{\star} &= \sum_i \sigma_i = \Vert \sigma\Vert_1 \\
\Vert A \Vert_F &= (\sum_i \sigma_i )^{\frac{1}{2}} = \Vert \sigma \Vert_2 \\
\Vert A \Vert_{2} &= \max_i \sigma_i = \Vert \sigma \Vert_{\infty}
\end{align}
$$


上面的$\Vert A \Vert_{\star}$表示矩阵的核范数，因为其本质上为奇异值的$L_1$范数。正如在鼓励稀疏性的优化中，我们用$L_1$范数来近似$L_0$范数。我们也用核范数（Nuclear Norm）来作为矩阵奇异值的非零元的个数（秩）的近似，达到鼓励低秩性质的优化。



对于Schatten-p范数，根据下面的定理，通常可以求解其次梯度的表达式，由于定理的证明较为复杂，我们直接给出结论，


$$
\begin{align}
\partial \Vert A \Vert & = conv(U DV^T),\text{ With } A = U \Sigma V^T , D =diag(\partial \Vert \sigma \Vert)
\end{align}
$$


也即其次梯度可以写成与奇异向量相关的凸组合，注意上式中$\Vert A \Vert$为Schatten-p范数，而$\Vert \sigma \Vert $为其对应的向量范数，两者并不相同。



根据上述结论，我们可以给出核范数的次梯度，对矩阵作奇异值分解SVD，并且将其奇异值按降序排列，并且按照非零奇异值和零奇异值对应分块，


$$
\begin{align}
\partial \Vert A \Vert_{\star} &= \sum_i \lambda_i U D V^T \\
&= \sum_i \lambda_i U_1 D_1 V_1^T + \sum_i \lambda_i U_2 D_2 V_2^T \\
&= \sum_i \lambda_i U_1 V_1^T + \sum_i \lambda_i U_2 D_2 V_2^T ,\text{With } D_1 = 1,D_2 = diag([-1,1]) \\
&=U_1 V_1^T + U_2 T V_2^T, \text{With } T = \sum_i \lambda_i D_2 \\
\end{align}
$$


再根据，


$$
\sigma_1(T) = \sigma_1(\sum_i \lambda_i D_2) \le \sigma_1(\sum_i \lambda_i) = 1
$$


得到最终的核范数的次梯度的表达式，
$$
\begin{align}
\partial \Vert A \Vert_{\star} = U_1 V_1^T + U_2 T V_2^T, \forall \sigma_1(T) \le 1 \\
\end{align}
$$
将该结果代入关于先前证明的关于矩阵次梯度的充要条件之后可以验证上式成立。



## Optimization

下面以$L_1$范数和核范数作为正则项为例，介绍和矩阵范数次梯度相关的优化。

### Sparse Regularization

通过$L_1$范数惩罚鼓励稀疏性，在线性回归模型中被称为Lasso回归，该思想在神经网络剪枝等也是常用的。

例如文章 [NIPS‘ DessiLBI: Exploring Structural Sparsity of Deep Networks via Differential Inclusion Paths](https://arxiv.org/pdf/2007.02010v1.pdf) 就利用类似的思想进行神经网络的优化。



考虑问题，


$$
\begin{align}
\min_Y \frac{1}{2}\Vert X - Y \Vert_F^2 +\lambda \Vert Y \Vert_1
\end{align}
$$


根据凸函数的次梯度条件


$$
\begin{align}
0 &= Y-X + \lambda sgn(Y) \\
Y &= X - \lambda sgn(Y) \\
\end{align}
$$


简单地分类讨论可以得到上述方程的解，


$$
\begin{align}
Y_{ij} &= 0 ,X_{ij} \in [-\lambda ,\lambda] \\
Y_{ij} &= X_{ij} -\lambda, X_{ij}> \lambda \\
Y_{ij} &= X_{ij} +\lambda, X_{ij}< -\lambda 
\end{align}
$$


可以看到$\lambda$作为一个惩罚因子，当$X_{ij}$地绝对值较小的时候将其置为0，从而获得解$Y$地稀疏性，而对于并未被置为0的$X_{ij}$。该正则项的作用也是令其向0靠拢。



### Low-Rank Regularization

采用核范数作为正则项，可以得到低秩的近似结果，另一种方法是使用矩阵低秩逼近的思路。

关于最佳低秩逼近，感兴趣的读者可以移步至 [特征值不等式与最佳低秩逼近](https://truenobility303.github.io/Low-Rank-Approximation/) 

这里介绍基于核范数惩罚的凸优化的解法，将问题转化为，


$$
\min_Y \frac{1}{2} \Vert X - Y \Vert_F^2 + \lambda \Vert Y \Vert_{\star}
$$
使用关于核范数的次梯度，可以验证其解为，


$$
Y = U (\Sigma - \lambda I)_+ V^T , X = U \Sigma V^T
$$


下面给出直观但较为简略的证明，根据次梯度，


$$
\begin{align}
\partial L &= Y - X + \lambda U_1 V_1^T + \lambda U_2 T V_2^T , \sigma_1(T) \le 1 \\
&=    U (\Sigma - \lambda I)_+ V^T - U \Sigma_ V^T + \lambda U_1 V_1^T + \lambda U_2 T V_2^T \\
&= U_1(\Sigma_1 -\lambda I) V_1^T -(U_1 \Sigma_1 V^T +U_2 \Sigma_2 V_2^T) + \lambda U_1 V_1^T + \lambda U_2 T V_2^T \\
&= \lambda U_2 T V_2^T - U_2 \Sigma V_2^T \\
&= O, \text{ Let } T =\lambda^{-1} \Sigma
\end{align}
$$


也即核范数惩罚相当于根据超参数$\lambda$, 将较小的奇异值都置为0，仅保留了较大的奇异值，这与低秩逼近也有异曲同工之妙。



## Application

上面介绍了两个加入矩阵范数正则的优化问题的解，本节主要介绍其应用。

### Robust PCA

鲁棒PCA类似于PCA的思想，鲁棒PCA认为信号由原始信号再加上噪声项扰动组成，而原始信号通常是低秩的，而噪声项符合稀疏特性，因此鲁棒PCA将问题转化为，


$$
\min \frac{1}{2} \Vert X - Y -Z \Vert_F^2 +\lambda \Vert Y \Vert_1 + \mu \Vert Z \Vert_{\star}
$$


鲁棒PCA和一般的PCA思想相近，但求解方式是完全不同的，关于PCA和概率PCA，可以移步至 [主成分分析PCA](https://truenobility303.github.io/PCA/)



对于该问题，简单的思路是采用轮换对称坐标下降法，也即固定某一些变量，对另外的变量进行优化，逐次迭代进行。

当$Y$固定的时候，问题转化为核范数惩罚的优化问题，而当$Z$固定的时候，问题转化为$L_1$范数惩罚的优化问题，从而轮换地进行求解。



### Matrix Complement

矩阵填充通常用于推荐系统，例如用户-电影打分矩阵，通常是稀疏是，矩阵填充的任务是利用已知的稀疏数据预测（填充）整个矩阵，我们假设最终的矩阵满足低秩性质，例如用户之间、电影之间通常有很多是相似的，因此其通常满足线性相关，问题被转化为，


$$
\min \frac{1}{2} \Vert X \odot A - Y \odot A \Vert_F^2 +\Vert Y \Vert_{\star}
$$


其中，$A_{ij} = 0 \text{ or } 1$ 表示不缺失的元素，上式用Hadamard乘积表示该部分元素应该尽可能接近.

可以首先将$X$的非确实元素设置为0，之后使用核范数惩罚的优化计算$Y$的一个解，然后利用$Y$中的元素对$X$进行填充，然后不断迭代下去，该思想类似于EM算法中对于隐变量的求解，关于EM算法，可以移步至 [EM算法](https://truenobility303.github.io/GMM-PLSA/) 

基于轮换坐标下降，通常可以寻找到局部最优值，而对于凸问题，局部最优值等价于全局最优值。因此上述的优化算法在某种意义下是有效的。
