---
title: '回归分析下'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---





线性回归模型的推广，从多重共线性的解决到岭回归模型的建立，线性自平稳器的留一交叉验证的公式，再生希尔伯特空间（RKHS）与核岭回归模型的建立，以及Box-Cox变换下和对率回归模型（Logistic Regression）的建立。



<!--more-->



本文内容紧承接自 [回归分析上](https://truenobility303.github.io/Rregression-First/)



## Multi-Collinearity 

线性回归模型中更重要的一个部分是对于多重共线性的检验，多重共线性指的是$X_1,...X_p$存在着线性相关或者强线性相关性的时候，此时由于 $rank(X^TX) = rank(X)$ , 如果$X$存在列线性相关，也即其存在某个特征可以被其他特征线性组合，那么该特征实际上是冗余的，检验的目的就是检测出上述多重共线性的情况。

最直接的想法是对$X_j$这个特征，用其余所有特征对其进行线性回归，并且利用其回归的$R_j^2$衡量多重共线性，此时可以得到方差膨胀因子的定义，


$$
\text{VIF}_j = \frac{1}{1-R_j^2}
$$


当回归效果很好的时候，例如 $R_j^2=1$， $\text{VIF}_j$ 趋近于正无穷，而当回归效果很差说明多重共线性很小的时候，此时$\text{VIF}_j$趋近于1. 因此上述的方差膨胀因子越大，说明多重共线性越强。

我们考虑对上述的方差膨胀因子进行计算，首先利用回归系数$R^2$ 的定义，最后一步根据 [高斯无向图模型](https://truenobility303.github.io/GGM/) 中分块矩阵求逆的公式, 可以推出方差膨胀因子和方差的关系，利用记号$X_{(j)}$表示除去$X_j$这一维度的特征后的其余特征构成的数据，


$$
\begin{align}
\text{VIF}_j &=  \frac{1}{1-R_j^2} \\
&= \frac{SST_j}{SSE_j} \\
&= \frac{SST_j}{X_{j}^T (I-H_j) X_{j}} \\
&= \frac{SST_j}{X_{j}^T (I-X_{(j)} (X_{(j)}X_{(j)})^{-1}X_{(j)}^T) X_{j}} \\
&= \frac{SST_j}{X_{j}^TX_{j} -X_{j}^TX_{(j)} (X_{(j)}X_{(j)})^{-1}X_{(j)}^T X_{j}} \\
&= SST_j(X^T X)_{jj}^{-1} \\
\end{align}
$$



利用上述式子可以计算得到方差膨胀因子，而如果想要知道其和方差的联系，


$$
\begin{align}
\text{VIF}_j &= SST_j(X^T X)_{jj}^{-1} = \frac{SST_j Var[\hat \beta_j]}{\sigma^2}\\ 
\end{align}
$$



## Ridge Regression

而对于存在上述多重共线性的情况，矩阵$X^TX$ 通常是不可逆的或者是很难求逆的，一个简单的想法是对其加上一个小的正数 $\lambda$ 使其便于求解，此时的估计量变成了，
$$
\hat \beta = (X^TX +\lambda I)^{-1} X^T Y
$$


下面我们将逐次证明，该方法不仅仅是一种数值计算层面的技巧，其背后蕴含着其深刻的道理。

上述方法就是著名的岭回归（Ridge Regression），一种理解的角度是从正则项的角度，实际上也相当于对参数$\beta$加上了一个高斯先验分布的假设后求解其最大后验估计，


$$
\begin{align}
\min \frac{1}{2}\Vert X \beta -Y \Vert_2^2 +\frac{1}{2} \lambda \Vert \beta \Vert_2^2 =\min \frac{1}{2}(X \beta -Y)^T (X \beta -Y) + \frac{1}{2}\lambda \beta^T \beta 
\end{align}
$$


对其求导求解其极值点可以得到，


$$
\begin{align}
X^T (X \beta -Y) + \lambda \beta = 0 \\
\hat \beta = (X^TX + \lambda I)^{-1}X^T Y
\end{align}
$$


观察岭回归的结果，如果对$X$采用奇异值分解，$X = U D V^T$, 可以得到，


$$
\begin{align}
\hat \beta &= V(D^2 +\lambda I)^{-1} D U^T Y \\
&= V diag(\frac{d_i}{d_i^2+\lambda}) U^T Y
\end{align}
$$


则$\lambda$的作用相当于对$X$的奇异值进行了压缩，使得最终求逆的结果（实际上也即对于最终的方差）进行了一个向0的压缩，因此我们可以想象到岭回归的结果会具有更小的方差，后面也将详细证明这一点。

为了严谨地论证上述猜想，需要对岭回归估计的期望和方差进行计算，


$$
\begin{align}
E[\hat \beta] &= (X^TX + \lambda I)^{-1}X^T X\beta \\
&= V^T (D^2+\lambda I)^{-1} D^2 V \beta \\
&= V^T diag(\frac{d_i^2}{d_i^2+\lambda}) V \beta\\
Var[\hat \beta] &= \sigma^2 (X^TX + \lambda I)^{-1}X^T X (X^TX + \lambda I)^{-1} \\
&= \sigma^2 V^T(D^2+\lambda I)^{-1} D^2(D^2+\lambda I)^{-1}V \\
&= \sigma^2 V^T diag(\frac{d_i^2}{(d_i^2 +\lambda)^2}) V
\end{align}
$$

类似地，经过计算可以发现岭回归得到的$\hat \beta$的向量模长变小了，这和方差缩减的性质也是等价的，岭回归相当于对$\hat \beta$起到了向0压缩作用。

经过上述计算可以清楚地看见，岭回归给出的结果虽然是一个有偏的结果，但其却起到了缩减估计方差的作用.

计算岭回归得到的偏差的模长，可以发现其虽然是有偏估计，但偏差存在上界，


$$
\begin{align}
\text{bias}(\hat \beta) &= E[\hat \beta] - \beta \\
&= (X^TX+\lambda I)^{-1} X^T X\beta  - \beta\\
&= (X^TX +\lambda I)^{-1} (X^TX - (X^TX+\lambda I)) \beta \\
&= -(X^TX + \lambda I)^{-1} \lambda \beta \\
\Vert \text{bias}(\hat \beta) \Vert_2^2 &= \Vert (X^TX + \lambda I)^{-1} \lambda \beta \Vert_2^2 \\
& \le \Vert (X^TX+ \lambda I)^{-1} \lambda \Vert_2^2 \Vert \beta \Vert_2^2 \\
&\le \Vert \beta \Vert_2^2
\end{align}
$$


因此可以认为岭回归以偏差作为代价，换取了更小的方差，根据方差-偏差分解，我们关注于岭回归是否可以降低回归的均方误差MSE，

$$
\begin{align}
MSE &= E \Vert \hat \beta -  \beta \Vert_2^2  \\
&= E \Vert \hat \beta - E \hat \beta + E \hat \beta - \beta \Vert_2^2  \\
&= E (\hat \beta - E \hat \beta)^T(\hat \beta - E \hat \beta) + ( E \hat \beta - \beta)^T (E \hat \beta - \beta) \\
&= trE[(\hat \beta - E \hat \beta)(\hat \beta - E \hat \beta)^T] +  ( E \hat \beta - \beta)^T (E \hat \beta - \beta) \\
&= trVar[\hat \beta] + \Vert \text{bias}(\hat \beta) \Vert_2^2
\end{align}
$$

将岭回归的结果代入上述偏差-方差分解，



$$
\begin{align}
MSE &= E \Vert \hat \beta -  \beta \Vert_2^2  \\
&= tr(\sigma^2 V^T diag(\frac{d_i^2}{(d_i^2 +\lambda)^2}) V) + \lambda^2 \beta^T V^T diag(\frac{1}{(d_i^2+\lambda)^2} )V \beta \\
&= \sigma^2 tr( diag(\frac{d_i^2}{(d_i^2 +\lambda)^2}) )+ \lambda^2 \beta^T V^T diag(\frac{1}{(d_i^2+\lambda)^2} )V \beta \\
\end{align}
$$

考虑$\lambda=0$附近的梯度，


$$
\begin{align}
\nabla_{\lambda=0} &= -2\sigma^2 diag(\frac{d_i^2}{(d_i^2 +\lambda)^3}) + 2\lambda  \beta^T V^T diag(\frac{1}{(d_i^2+\lambda)^2} )V \beta - 2 \lambda^2 \beta^T V^T diag(\frac{1}{(d_i^2+\lambda)^3} )V \beta \\
&= -2\sigma^2 diag(\frac{d_i^2}{(d_i^2 +\lambda)^3}) <0
\end{align}
$$


因此至少在$\lambda=0$附近，岭回归一定是可以起到降低MSE的作用的。

---

为了对岭回归的结果进行进一步的分析，考虑一类特殊的情况，此时$X^TX=I$ 也即$X$为一个正交矩阵，



$$
\begin{align}
E[\hat \beta] &= V^T diag(\frac{d_i^2}{d_i^2+\lambda}) V \beta =\frac{1}{1+\lambda} \beta  \\ 
Var[\hat \beta] &= \sigma^2 V^T diag(\frac{d_i^2}{(d_i^2 +\lambda)^2}) V = \frac{\sigma^2}{(1+\lambda)^2} 
\end{align}
$$



更进一步代入偏差-方差分解的公式，


$$
\begin{align}
MSE &= \frac{\sigma^2 p}{(1+\lambda)^2} + \frac{ \lambda^2 \beta^T \beta }{(1+\lambda)^2} \\
\frac{d MSE}{d \lambda}  &= -\frac{2 \sigma^2 p}{(1+\lambda)^3} + \frac{2 \lambda \beta^T \beta}{(1+\lambda)^3} = 0 
\end{align}
$$


可以解得使得MSE最小的点，记作$\lambda_0$,


$$
\lambda_0 = \frac{\sigma^2 p}{\beta^T \beta}
$$


而MSE随着$\lambda$的增大呈现先减小后增大的趋势，当达到某一个位置岭回归对MSE的效果将和普通线性回归一致，在该点之后再增加正则项系数$\lambda$ 将起到适得其反的效果，将该点记作 $\lambda_{max}$,其应该满足，


$$
\lambda_{max} = \frac{2 \sigma^2 p }{\beta^T \beta - \sigma^2 p} = 2 +\frac{2}{1-\lambda_0}
$$


观察上面的结果可以发现其和$\Vert \beta \Vert_2^2, \sigma^2$ 相关，其意义从信号处理的角度理解正好对应着信噪比，当信噪比越低的时候，噪声越多，此时岭回归起到的作用将更有效。



## Cross Validation

本节考虑将普通线性回归中的模型选择方法推广到岭回归中，  同样利用AIC进行模型选择，首先需要利用矩阵的迹推广AIC中关于模型复杂度的量，可以得到相应的AIC准则，


$$
\begin{align}
\text{AIC} &= -\frac{n}{2} \log SSE - tr(X(X^TX+\lambda I)^{-1}X^T) \\
&= -\frac{n}{2} \log SSE - tr(H), \text{With } H = X(X^TX+\lambda I)^{-1}X^T
\end{align}
$$

为了选择好的超参数$\lambda$, 常见的做法是使用交叉验证，例如K-折交叉验证，而特殊的情况是留一交叉验证，其好处是其之可以用公式计算得到，考虑用除掉样本$X_i$以外的所有数据作为训练集，而采用$X_i$作为测试集，计算平方误差损失，利用$X_{(i)}$表示去除该样本剩下的样本，

$$
\begin{align}
(x_i \hat \beta_{(i)} - y_i)^2 &= (x_i^T (X_{(i)}^TX_{(i)}+\lambda I)^{-1}X_{(i)}^T y_i - y_i)^2\\
&=(x_i^T A^{-1} X_{(i)}^T y_i- y_i)^2, \text{Let } A = X_{(i)}^TX_{(i)}+\lambda I
\end{align}
$$

为了求解所需要的项，利用Woodbury公式，


$$
\begin{align}
(X^TX+\lambda I)^{-1} &= (\sum_{j=1}^n x_j x_j^T+\lambda I)^{-1} \\
&= (\sum_{j\ne i}^n x_j x_j^T+ x_ix_i^T+\lambda I)^{-1} \\
&= (A +x_ix_i^T)^{-1} \\
&= A^{-1}  - A^{-1} x_i (1+x_i^T A^{-1}x_i)^{-1} x_i^TA^{-1} \\
\end{align}
$$


因而我们可以知道，


$$
\begin{align}
x_i^T (X^TX+\lambda I)^{-1} &= x_i^T A^{-1} - x_i^T A^{-1} x_i(1+x_i^T A^{-1}x_i)^{-1} x_i^TA^{-1}  \\
&=(1-x_i^T A^{-1} x_i(1+x_i^T A^{-1}x_i)^{-1}) x_i^T A^{-1} \\
&= (1-\frac{x_i^T A^{-1}x_i} {1+x_i^T A^{-1}x_i} )x_i^T A^{-1} \\
&= \frac{1}{1+x_i^T A^{-1}x_i} x_i^T A^{-1} \\
x_i^T A^{-1} &= (1+x_i^T A^{-1}x_i) x_i^T (X^TX+\lambda I)^{-1} \\
x_i^T A^{-1}x_i &= (1+ x_i^T A^{-1}x_i) x_i^T (X^TX+\lambda I)^{-1}x_i \\
x_i^T A^{-1}x_i &= \frac{x_i^T (X^TX+\lambda I)^{-1}x_i}{1-x_i^T (X^TX+\lambda I)^{-1}x_i} \\
x_i^T A^{-1} &= \frac{x_i^T (X^TX+\lambda I)^{-1}}{1-x_i^T (X^TX+\lambda I)^{-1}x_i}  \\
\end{align}
$$


最终化简得到，


$$
\begin{align}
(x_i^T \hat \beta_{(i)} - y_i)^2 
&=(x_i^T A^{-1} X_{(i)}^T y_i- y_i)^2 \\
&=(x_i^T A^{-1} (X^T -x_i)y_i- y_i)^2 \\
&=(\frac{x_i^T (X^TX+\lambda I)^{-1}(X^T-x_i)y_i}{1-x_i^T (X^TX+\lambda I)^{-1}x_i}  - y_i)^2 \\
&= (\frac{x_i^T (X^TX+\lambda I)^{-1}X^Ty_i-y_i}{1-x_i^T (X^TX+\lambda I)^{-1}x_i} )^2 \\
&= \frac{(x_i^T \hat \beta - y_i)^2}{(1-H_{ii})^2}, \text{With } H = X^T (X^TX+\lambda I)^{-1}X \\
\end{align}
$$



上面的推导稍微复杂了一点，但推导中用到的求解矩阵方程的技术仍不失为一种好的技巧，下面利用另一边的Woodbury公式，给出一个稍微简单一点的证明，


$$
\begin{align}
x_i^T \hat \beta_{(i)} - y_i &= x_i^T( (X_{(i)}^T X_{(i)} + \lambda I)^{-1}X_{(i)}^Ty_i) - y_i \\
&= x_i^T( (X^TX + \lambda I - x_ix_i^T)^{-1}X_{(i)}^Ty_i) - y_i \\
&= x_i^T( (X^TX + \lambda I - x_ix_i^T)^{-1}(X^Ty_i - x_i y_i) - y_i \\
&= x_i^T( (K - x_ix_i^T)^{-1}(X^Ty_i - x_i y_i) - y_i ,\text{Let } K = X^TX+\lambda I \\
&=x_i^T (K^{-1}+K^{-1} x_i(1- x_i^T K^{-1} x_i)^{-1}x_i^TK^{-1})(X^Ty_i - x_i y_i) - y_i , \text{By Woodbury Formula} \\
&=x_i^T (K^{-1}+\frac{K^{-1} x_ix_i^TK^{-1}}{1-x_i^TK^{-1}x_i })(X^Ty_i - x_i y_i) - y_i \\
&= \frac{x_i^TK^{-1}}{1-x_i^T K^{-1}x_i}(X^Ty_i - x_i y_i) - y_i \\
&= \frac{x_i^TK^{-1}X^Ty_i - x_i^T K^{-1} x_i y_i}{1-x_i^T K^{-1}x_i} - y_i \\
&= \frac{x_i^TK^{-1}X^Ty_i - y_i}{1-x_i^T K^{-1}x_i} \\
&= \frac{x_i^T \hat \beta - y_i}{1-H_{ii}}
\end{align}
$$




如果当$p \rightarrow \infty$的时候，根据大数定理，利用矩阵的迹的均值近似矩阵的对角元素，可以得到广义的留一交叉验证的公式，

$$
(x_i \hat \beta_{(i)} - y_i)^2  = \frac{(x_i^T \hat \beta - y_i)^2}{(1- \frac{tr(H)}{p})^2}
$$


遍历所有的$i$，取上述平方损失的平均值，可以得到平均的留一交叉验证的损失，根据该损失就可以选取最优的$\lambda$ 


---

实际上，上述的结论具有一个更普遍意义上的结论，对于一个自稳定的线性平滑器，其留一交叉验证都具有上述的公式，我们以线性回归模型为例详解自稳定线性平滑器的含义。

线性指的是预测值可以写成训练集的线性组合，而自稳定性指的是将拟合后的数据加入训练集重新训练，得到的结果与原数据集得到的结果一致，通过对线性回归模型的推导可以更加直观地理解上述定义的含义, 由于线性回归模型等价于最小化残差平方和，因此显然满足自稳定性。


$$
\begin{align}
\hat Y &= H Y ,\text{By Linear Property}\\
\tilde Y_i &= x_i \hat \beta_{(i)} = (H \tilde Y)_i = H_{ii} \tilde Y_i + \sum_{j \ne i} H_{ij} Y_j ,\text{By Self-Stable Property}\\
\hat Y_i &= x_i \hat \beta_i = (HY)_i = H_{ii} Y_i + \sum_{j \ne i} H_{ij} Y_j \\
\tilde Y_i &= \frac{\hat Y_i - H_{ii} Y_i}{(1-H_{ii})} \\
\tilde Y_i - Y_i &=  \frac{\hat Y_i - H_{ii} Y_i}{1-H_{ii}} - Y_i = \frac{\hat Y_i - Y_i}{1-H_{ii}}  
\end{align}
$$



经过简单的推导就得到了上面的结论，该结论更为普遍，且推导过程也更为简洁。



## Kernel Ridge Regression

核岭回归，将核方法引入岭回归的估计中, 相当于在一个高维空间内进行岭回归操作。

在岭回归的基础上使用矩阵求逆的公式进行变换，并且注意到 $XX^T$ 本质上相当于元素之间的内积，如果我们知道在高维空间中的内积以及内积对应的表示矩阵为 $K_{n \times n}$, 利用该矩阵可以在高维空间内进行岭回归，其好处是仅仅需要知道核函数，而不需要真正知道低维空间到高维空间的映射。


$$
\begin{align}
\hat Y &= X(X^T X+ \lambda I)^{-1} X^TY \\
&= XX^T(XX^T+ \lambda I)^{-1} Y \\
&= K(K+\lambda I)^{-1}Y ,\text{Let } K  = XX^T
\end{align}
$$


对于核方法的简单介绍和性质可以参见 [核方法](https://truenobility303.github.io/PCA/) ，根据核函数的性质，且我们知道 加法、Hadamard积、Kronecker积都保持核函数的正定性，我们可以证明如下的几个常见的核函数都满足正定性，



首先，如下定义的多项式核满足正定性质，


$$
\begin{align}
k(x,y) &= \langle x, y \rangle \text{ positive definite } \\
k(x,y) &= \langle x, y \rangle +c \text{ positive definite ,with Addition} \\
k(x,y) &= (\langle x, y \rangle +c)^p \text{ positive definite ,with Hadamard Product} \\
\end{align}
$$


再者，如下定义的高斯核也满足正定性质，


$$
\begin{align}
\exp(x) &= \sum_{k=0}^{\infty} \frac{x^k}{k!} \\
k(x,y) &= \langle x , y \rangle \text{ positive definite} \\
k(x,y) &= \exp( -\frac{\Vert x - y \Vert^2}{2\sigma^2}) \\
&=\exp(-\frac{\Vert x \Vert^2}{2\sigma^2}) \exp(- \frac{\Vert y \Vert^2}{2\sigma^2}) \exp(\frac{\langle x,y \rangle}{\sigma^2}) \text{ postive definite, With Addtion and Hadamard Product} 
\end{align}
$$


### Regression in RKHS

为了更加深刻地理解核岭回归，我们在再生Hilbert空间（RKHS，Reproducing Kernel Hilbert Space）中考虑该问题，通过本节的内容可以更加深入地理解核岭回归和一般的线性回归的联系。

首先对正定核做谱分解，并且利用谱分解的结果定义内积


$$
\begin{align}
k(x,x') &= \sum_{j=1}^{\infty} \gamma_i \varphi_j(x ) \varphi(x') \\
\langle g ,g' \rangle &= \sum_{j=1}^{\infty} \frac{\beta_j \beta_j'}{\gamma_j}, \text{Let } g(x) = \sum_{j=1}^{\infty} \beta_j \varphi_j(x)
\end{align}
$$


在上述的基础上，可以得到一些有用的性质，证明只需要进行基展开并且代入上述定义即可，也可以参见 [Blog](https://truenobility303.github.io/PCA/),这些性质和再生性密切相关。


$$
\begin{align}
\langle k( \cdot,x) , g \rangle &= g(x) \\
\langle k(\cdot,x), k(\cdot,y) &= k(x,y) \\
\end{align}
$$


下面的表示定理（Representer Theorem）刻画了核岭回归的本质，


$$
\begin{align}
f &= \text{argmin}_f \sum_{i=1}^n \mathcal{L}(Y_i,f(X_i)) + P(\Vert f \Vert) \\
&= \sum_{i=1}^n \alpha_i k(\cdot,X_i) \\
\end{align}
$$


其中 $\mathcal{L}$ 为损失函数，而 $P$ 为一个单调函数作为正则项存在,证明中利用了子空间的正交分解，


$$
\begin{align}
f &= \sum_{i=1}^n \alpha_i k(\cdot,X_i) + r,\text{With } \langle r,k(\cdot, X_i) \rangle = 0 \\
&= g+ r ,\text{Let } g = \sum_{i=1}^n \alpha_i k(\cdot,X_i) \\
f(X_i) &= g(X_i) + r(X_i) = g(X_i) + \langle r, k(\cdot, X_i) \rangle =g(X_i) \\
\Vert f \Vert^2 &= \Vert g + r \Vert^2  = \Vert g \Vert^2 + \Vert r \Vert^2 \\

\end{align}
$$


在上面的基础上，可以证明当且仅当解为函数 $g$ 的时候取到总损失函数的极小值，


$$
\begin{align}
\sum_{i=1}^n \mathcal{L}(Y_i,f(X_i)) + P(\Vert f \Vert) & \ge \sum_{i=1}^n \mathcal{L}(Y_i,g(X_i)) + P(\Vert g \Vert) \\
\end{align}
$$


该表示定理虽然推导简单，但其对于无限维空间上的回归是重要的结论，其证明掠对于无限维空间上的学习问题，其解可以被有限维表示，并且表示与核（Kernal）密切相关。

而核岭回归实际上是上述定理的一个特例，对应与损失函数取平方损失，正则项取 $L_2$ 范数的情况，


$$
\begin{align}
L &= \sum_{i=1}^n( f(X_i) - Y_i)^2 + \lambda \Vert f \Vert^2 \\
\end{align}
$$
根据定理我们知道上式的最优解为 $f=g$ ,


$$
\begin{align}
f &= g=\sum_{i=1}^n \alpha_i k(\cdot, X_i) \\
f(X_i) &= \langle k(\cdot, X_i) , f \rangle = K \alpha \\
\Vert f \Vert^2 &= \langle f , f\rangle  = \alpha^T K \alpha 
\end{align}
$$

关于 $\alpha$ 求解其最优的系数就可以得到岭回归的结果，



$$
\begin{align}
L &=  \Vert K \alpha-y \Vert^2 + \lambda \alpha^T K \alpha \\
\frac{dL}{d \alpha} &= 2(K^T(K \alpha - y) + \lambda K \alpha) = 0\\ 
K^T y &=K^T K \alpha + \lambda K \alpha & \\
\alpha &= (K^TK+ \lambda K)^{-1} K^T y \\
&=(K+ \lambda I)^{-1} y ,\text{With } K = K^T, K \succeq 0 \\
\hat y &= K \alpha = K(K+\lambda I)^{-1} y
\end{align}
$$




## Generalized Regression

本节介绍更为广义的回归模型



### Box-Cox Transformation

Box-Cox变换是将非线性回归问题转化为线性回归问题的一种常用方式，其变换定义为，


$$
\begin{align}
y^{(\lambda)} &= \frac{y^{\lambda}- 1}{\lambda}, \lambda \ne 0 \\
y^{(\lambda)} &= \log y, \lambda = 0 \\
\end{align}
$$
其中 $\lambda$ 为设定的参数，当 $\lambda=0$  的时候正好为对数线性回归模型，下面介绍如何使用极大似然估计 MLE选取 $\lambda$ 的过程，


$$
\begin{align}
L(\beta, \sigma^2, \lambda) &= \prod_{i=1}^n\frac{1}{\sqrt{2 \pi \sigma^2}} \exp(-\frac{y_i^{(\lambda)}-x_i \beta}{2\sigma^2}) \det J, \text{With} \det J = \prod_{i=1}^n y_i^{\lambda -1} \\
\log L(\beta, \sigma^2,\lambda) &=-\frac{n}{2} \log (2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \Vert y^{(\lambda) } - X \beta \Vert^2 + \log \det J \\
\end{align}
$$


可以看到前两项与 $\lambda $ 无关，利用普通线性回归的结论，可以得到，


$$
\begin{align}
\hat \beta &= (X^TX)^{-1}X^T y^{(\lambda)} \\
\hat \sigma^2 &= \frac{1}{n} \Vert y^{(\lambda)} - X \hat \beta \Vert^2 \\
\log L(\lambda) &= -\frac{n}{2} \log 2 \pi- \frac{n}{2} \log \hat \sigma^2 + \log \det J \\
&=-\frac{n}{2} \log 2 \pi- \frac{n}{2} \log \frac{\text{SSE}}{n}  + \log \det J \\ 
&=-\frac{n}{2} \log \text{SSE} + \log \det J + k, \text{ const } k \\
\max \log L(\lambda) &= \max[ -\frac{n}{2} \log \text{SSE} + \log \det J ] \\
&= \max[ -\frac{n}{2} \log \text{SSE} + \frac{n}{2} \log (\det J)^{\frac{2}{n}} ] \\
&= \min [\log \text{SSE} - \log (\det J)^{\frac{2}{n}}] \\
&=\min \log \frac{\text{SSE}}{(\det J)^{\frac{2}{n}}} \\
&=\min \frac{y^{(\lambda)} (I-H) y^{(\lambda)}}{(\det J)^{\frac{2}{n}}} ,\text{Let } H = X(X^TX)^{-1}X^T\\
&= \min \ \tilde y (I-H) \tilde y, \text{Let } \tilde y = \frac{y^{(\lambda)}}{(\det J)^{\frac{1}{n}}}
\end{align}
$$


也即Box-Cox变换下的 $\lambda$ 选择相当于也是在最小化某种意义下的残差平方和。



### Logistic Regression

线性回归解决自变量为定性变量，因变量也为定性变量的回归问题，而Logistic 回归可以解决自变量为定性变量，而因变量为定量变量的回归问题，此处我们考虑因变量为0-1变量的情况，对于非0-1变量的情况，可以类似Logistic回归的情况加以推广。


$$
\begin{align}
y_i & \sim \text{Bernoulli}(p_i) \\
p_i &= \sigma(x_i \beta) =\frac{\exp x_i \beta }{1+ \exp x_i \beta} \\
x_i \beta  &= \text{logit}(p_i) = \log \frac{p_i}{1-p_i} 
\end{align}
$$

本质上Logistic回归是一种广义线性回归模型，利用Sigmoid函数建立起自变量和因变量之间的联系，下面我们研究Sigmoid函数的性质，


$$
\begin{align}
\sigma(-x) &= 1- \sigma(x) \\
\sigma'(x) &= \sigma(x)(1- \sigma(x))
\end{align}
$$



利用上述性质，采用极大似然估计求解参数 $\beta$  的估计，但由于该问题中不存在显示解，采用Newton迭代法进行，

可以看到此时极大似然估计也等价于最小化交叉熵损失，



$$
\begin{align}
l &=\log L \\
&= \log \prod_{i=1}^n p_i^{y_i} (1-p_i)^{1-y_i} \\
&=\sum_{i=1}^n y_i \log p_i + (1-y_i) \log (1- p_i) \\
&=\sum_{i=1}^n y_i \log \frac{p_i}{1-p_i} + \log (1- p_i) \\
&=\sum_{i=1}^n y_i x_i \beta  + \log (1- \sigma(x_i \beta)) \\
\nabla l &= \sum_{i=1}^n y_ix_i - \frac{\sigma'(x_i \beta )}{1-\sigma(x_i \beta)} \\
&= \sum_{i=1}^n y_ix_i - \frac{\sigma'(x_i \beta )}{1-\sigma(x_i \beta)} \\
&= \sum_{i=1}^n y_ix_i - x_i \sigma(x_i\beta) \\
&=\sum_{i=1}^n y_ix_i - x_i p_i \\
&= X^T(y- p) \\
\nabla^2 l &= - \sum_{i=1}^n x_i x_i^T \sigma(x_i \beta) (1-\sigma(x_i \beta)) \\
&= - \sum_{i=1}^n x_i x_i^T p_i(1-p_i) \\
&= - X^T DX, \text{Let } D = \text{diag}(p_i(1-p_i))
\end{align}
$$



最终代入Newton法得到，
$$
\begin{align}
\hat \beta^{(t+1)} &= \hat \beta^{(t)} - (\nabla^2 l)^{-1} \nabla l \\
&= \hat \beta^{(t)} + (X^T DX)^{-1} X^T(y-p) \\
\end{align}
$$


上式和带权最小二乘非常相近，注意到其相当于用下面的关系式做了归一化的带权最小二乘，


$$
\begin{align}
E[y] &= p \\
Var[y] &= \text{diag}(p(1-p)) = D
\end{align}
$$









