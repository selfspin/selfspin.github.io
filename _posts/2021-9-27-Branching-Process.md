---
title: '分支过程'
toc: true
excerpt_separator: <!--more-->
tags:
  - 随机过程
---

分支过程是一种特殊的马尔可夫链，但由于其比较特殊且在显示生活中很常见，单独考虑之。分支过程通常用于分析生物繁衍、粒子分裂等问题。

<!--more-->

由于分支过程为特殊的马尔可夫链，关于马尔可夫链，可以参见 [马尔可夫链](https://truenobility303.github.io/Markov-Chain/)



## Distribution of Branching Process

定义$X_n$为第$n$代的数量，$X_1$ = 1,每个个体繁衍的后代个数为随机变量$Z$, 则$X_n = \sum_{i=1}^{X_{n-1}} Z_i$

为了研究$X_n$的分布，引入生成函数，$\phi(s) = E[s^X] = \sum_{k=1}^{\infty} s^k P(X=k)$

可以得到，$\phi(0) = 0, \phi(1) = E[X] $

$\phi^{(n)}(s) = \sum_{k=n}^{\infty} k! s^{n-k} P_n(X=k), P_n(X=k)=\frac{\phi^{(k)}(0)}{k!}$ 

类似于特征函数，生成函数和分布函数也一一对应。

下面，求$X_n$的生成函数：

$\phi_n(s) = E[s^{X_n}] =  E[s^{\sum_{i=1}^{X_{n-1}} Z_i}] = E[ \prod_{i=1}^{X_n-1} s^{Z_i} ] = E[\phi(s)^{X_{n-1}}] = \phi_{n-1}(\phi(s))$

根据，$X_1 = Z$ ,可以递推得到，$\phi_n(s) = \phi^{(n)}(s)$ 

---

类似地，利用条件期望和条件方差公式，可以计算$E[X_n], Var[X_n]$

首先，根据$Z_i(i.i.d)$的假设，设

$E[X_n] = E[\sum_{i=1}^{X_{n-1}} Z_i] = E[X_{n-1} \mu] = \mu E[X_{n-1}]$, 递推可以得到，$E[X_n] = \mu^n$

$Var[X_n] = Var[E[\sum_{i=1}^{X_{n-1}} Z_i] +E[Var[\sum_{i=1}^{X_{n-1}}Z_i]] =Var[X_{n-1}\mu] + E[X_{n-1} \sigma^2] = \mu^2 Var[X_{n-1}] + \sigma^2 E[X_{n-1}]$

同样可以得到，$Var[X_n]$的递推公式。

## Extinction Probability

我们希望研究一个分支过程是否会走向灭绝，定义$\tau_n$为第$n$代灭绝的概率。

显然地，$\tau_{n+1} \ge \tau_n$ , 且灭绝概率的上界为1，单调有界数列必有极限，该极限就是灭绝概率，定义为$\tau$

$\tau = \lim_{n \rightarrow \infty} P_n(X=0) = \phi^{(n)}(0)$

根据，$\phi^{(n+1)}(0) = \phi (\phi^{(n)}(0))$ ,两端同时取极限，可以得到,$\tau = \phi(\tau)$

观察$\phi(\tau)$的性质，$\phi(0) = p_0,\phi(1) = 1, \phi'(1)= \mu,  \phi''(t) \ge 0$

可以得到下面的结论：

* $p_0 = 0,\tau=0$
* $p_0 = 1,\tau =1$
* $0<p_0<1,\mu > 1, \tau = \phi(\tau)$
* $0<p_0<1,\mu \le 1, \tau =1$



