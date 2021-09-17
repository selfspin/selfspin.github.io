---
title: '常用分布与假设检验'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计

---

总结常用的分布和常见的假设检验方法。

<!--more-->

## 极限定理



定义随机变量$X$的特征函数，$M(t) = E[e^{itX}]$

求导可以得到，$M'(0) = E[X], M''(0) = -E[X^2]$.

对比傅里叶变换的公式，可以发现，特征函数相当于进行了一个傅里叶变换，同样有卷积定理，$M_{X+Y}(t) = M_X(t)M_Y(t)$

对于傅里叶变换的卷积定理的证明，可参见[卷积定理的证明](https://blog.csdn.net/xxmy7/article/details/109358114)

根据正态分布的表达式，算出其特征函数为$M(t) = e^{\mu t -  \frac{\sigma^2 t^2}{2}}$

根据特征函数，可以给出极限定理的简证。

其中关键是在于特征函数的Taylor展开，$M(Y) = M(0)+M'(0)t+M''(0)t^2 +o(t^2) =1+E[Y]t-E[Y^2]t^2$ 



### 大数定理

$Y = \frac{X}{n},M(\sum_iY_i) = (M(Y))^n$

显然，$E[Y] = \frac{\mu}{n}$

根据Taylor展开，$M(Y) =1+E[Y]t+o(t)=1+\frac{\mu}{n}t+o(t)$ 

$(M(Y))^n = (1+ \frac{\mu}{n}t+o(t))^n \rightarrow e^{\mu t},n\rightarrow\infty$. 

因此，$ \sum_i Y_i \rightarrow \mu$, 此即为大数定理。



### 中心极限定理

$Y = \frac{X-\mu}{\sigma \sqrt{n}}$

显然，$E[Y] = 0,E[Y^2] = \frac{1}{n}$

根据Taylor展开，$M(Y) =1+E[Y]t-E[Y^2]t^2+o(t^2)=1-\frac{1}{n}t^2+o(t^2)$ 

$M(\sum_i Y_i) = (M(Y))^n = (1-\frac{1}{n}t^2+o(t^2))^n \rightarrow e^{-\frac{1}{2}t^2},n \rightarrow \infty$

因此，$\sum_i Y_i \rightarrow \mathcal{N}(0，1)$，此即为中心极限定理



## 常用分布

大数定理可以理解为在样本量很大的时候，对均值的刻画，则中心极限定理可以理解为对方差的刻画。

我们发现，极限定理都与正态分布有关。根据中心极限定理，任何分布当其抽样个数足够多的时候，都服从正态分布。

因此，正态分布成为最常用的概率分布。

首先定义常用分布，后面将逐一说明其作用，

* 正态分布，$\mathcal{N}(\mu,\sigma^2)$
* 卡方分布，$\chi(n)= \sum_{i=1}^n X_i^2,X_i \sim \mathcal{N}(0,1)$
* t分布，$t(n) = \frac{X}{S / \sqrt{n}},X\sim\mathcal{N}(0,1),S^2 \sim \chi^2(n)$
* F分布，$F(n_1,n_2) = \frac{S_{1}^2 / n_1}{S_2 / n_2},S_1^2 \sim \chi(n_1), S_2^2 \sim \chi(n_2)$

### 正态分布

正态分布有很多良好的性质。

#### 性质1.1 独立的正态分布的线性组合仍然为正态分布

$X_i \sim \mathcal{N}(\mu_i, \sigma_i^2),\sum_i X_i \sim \mathcal{N}(\sum_iu_i,\sum_i \sigma_i^2)$

证明，根据特征函数即可，本质在于正态分布的特征函数为指数的形式。

#### 性质1.2 正态分布独立等价于不相关

使用联合特征函数，$M(t_1,t_2,...t_n) = E[e^{i\sum_i t_i X_i}]$

令$Y= \sum_i t_i X_i,E[e^{i\sum_i t_i X_i}]=E[e^{iY}]$,

$Y$同样符合正态分布，因此,$E[e^{iY}]=e^{E[Y]-D[Y]^2/2}=e^{\sum_i \mu_i t_i -\sum_{ij}t_i t_jCov[X_i,X_j]/2}$

从上式可见，当$X_i$不相关的时候，$Cov[X_i,X_j]=0$,因此，$M(t_1,t_2,..t_n) = M(t_1)M(t_2)...M(t_n)$，也即$X_i$相互独立。

#### 性质1.3 独立正态分布的样本均值和样本方差相互独立

根据性质1.1，可以知道样本均值，$\bar{X} = \sum_i X_i /n \sim \mathcal{N}(\mu,\sigma^2/n)$

定义样本方差，$S^2=\sum_i (X_i - \bar{X})^2/(n-1)$

分母为$n-1$的原因是为了保证无偏性，下面简证，

首先，$(n-1)S^2= \sum_i(X_i-\mu)^2-((\bar{X}-\mu) / \sqrt{n})^2 $

因此，$E[(n-1)S^2] =E[\sum_i(X_i-\bar{X})^2]=(n-1)\sigma^2,E[S^2] = \sigma^2$



只需要证明，$Cov[X_i- \bar{X},\bar{X}]=0$, 证明较为简单，

$X_i - \bar{X} = \sum_k I[k=i] - X_k /n$, $\bar{X} = \sum_k X_k /n$

因此，$Cov[X_i-\bar{X},\bar{X}]=D[X_k] \sum_k (I[k=i]-\frac{1}{n})\frac{1}{n}=0$

又根据性质1.2，正态分布不相关即为独立，因此$S$与$\bar{X}$也独立。



### 卡方分布

在假设检验中，卡方分布通常用于检验分布的方差。

#### 性质2.1 独立的卡方分布之和仍为卡方分布

该性质是显然的，根据卡方分布的定义，将其看作独立的正态分布之和即可，

#### 性质2.2 $(n-1)S^2 / \sigma^2 \sim \chi(n-1)$

根据$(n-1)S= \sum_i(X_i-\mu)^2-((\bar{X}-\mu) / \sqrt{n})^2 $

可以发现,$(n-1)S^2 / \sigma^2 \sim \chi(n-1)$

#### 性质2.3 卡方分布的均值和方差

$X \sim \chi(n),E[X] = n, D[X] =2n$ 

证明用到了正态分布的偶数阶矩，可以参见 [知乎回答](https://www.zhihu.com/question/293696778/answer/487409760)



### t分布

在假设检验中，当样本的均值和方差都未知时，需要引入t分布，依据是下面的性质。

#### 性质3.1 $ \bar{X} -\mu /(n-1)S \sim t(n-1)$

$(\bar{X}-\mu) / \sigma \sim \mathcal{N}(0,1)$,且$(n-1)S / \sigma \sim \chi(n-1)$,得证。



### F分布

F分布通常用于检验两个分布的方差，依据是如下性质

#### 性质4.1 $S_1^2 / S_2^2 \sim F(n-1,n-1)$

显然，$S^2 \sim \chi^2(n-1)$ ,根据定义即可知道。



## 假设检验

### 单分布均值检验

方差已知时，使用正态分布，$\frac{\bar{X}-\mu} { \sigma / \sqrt{n}} \sim\mathcal(0,1)$

方差未知时，需要使用t分布，$\frac{\bar{X} - \mu}{S / \sqrt{n}}  \sim t(n-1)$



### 单分布方差检验

无论均值是否已知，都使用卡方分布,区别是卡方分布的系数n

均值已知时，$\sum_i (X_i-\mu)^2 /\sigma^2\sim \chi^2(n)$

均值未知时，$(n-1)S /\sigma^2\sim \chi^2(n-1)$



### 双分布均值检验

需要检验两组样本的均值是否相等。

当两组样本的方差都已知的时候，使用正态分布即可。

$\bar{X}  - \bar{Y} / \sqrt{\sigma_x^2/n_x+ \sigma_y^2 /n_y} \sim \mathcal{N}(0,1)$



而当两组样本的方差未知的时候，需要两组样本的方差相等，使用t分布进行检验，

$ \frac{\bar{X} - \bar{Y}} {S_w \sqrt{1/n_x+1/n_y}} \sim t(n_x+n_y-2),S_w = \sqrt{\frac{(n_x-1)S_x^2+(n_y-1)S_y^2}{ (n_x+n_y-2)}}$

使用$S_w$而不使用单独的$S_x$或者$S_y$的原因是此时的$S_w$会具有更小的方差,根据卡方分布的方差计算即可知道。



### 双分布方差检验

不管均值是否已知，都使用F分布进行假设检验，

均值已知时，$\frac{\sum_i (X_i - \mu_x)^2 /n_x}{\sum_i (Y_i -\mu_y)^2 / n_y}\sim F(n_x,n_y)$

均值未知时， $S_x^2 / S_y^2 \sim F(n_x-1,n_y-1)$



### 单因素方差分析

在方差分析中，同样使用F分布进行检验。

方差分析中，假设有$s$个不同的因素，每个因素产生一组正态分布样本。

假设为所有因素产生的正态分布的均值相等。

定义组间离差和$S_A$，总离差和$S_T$，随机误差离差和$S_E$，则展开可得$S_T = S_A + S_E$. 

不管假设成不成立，都有$S_E \sim \chi^2(n-s)$

当假设成立的时候，$S_A \sim \chi^2(s-1)$

因此，在这个假设检验中，可以取检验统计量$S_A / S_E \sim F(s-1,n-s)$

