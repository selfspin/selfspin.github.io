---
title: 'The Ideal Intepolator'
excerpt_separator: <!--more-->
toc: true
tags:
  - 线性回归
---



Paper Reading: Harmless interpolation of noisy data in regression

<!--more-->





本文仅仅作为原论文的思想的简要引入。考虑经典的正态线性回归模型：


$$
\begin{align*}
y \sim x^\top  \beta  + \epsilon, \quad x \in \mathbb{R}^{p \times 1}\sim \mathcal{N}(0,\Sigma), \quad \epsilon \in \mathbb{R} \sim \mathcal{N}(0,\sigma^2).
\end{align*}
$$




抽取数据集 $(X,Y)$ , 其满足


$$
\begin{align*}
Y = X \beta + W, \quad W \in \mathbb{R}^{n \times 1} \sim \mathcal{N}(0,I_n).
\end{align*}
$$
 

将 $X$ 标准化后得到协方差为单位矩阵的白化数据集$Z$, 其满足

 
$$
\begin{align*}
Z = \Sigma^{-1/2} X, \quad Z_{ij} \sim \mathcal{N}(0,1).
\end{align*}
$$


并且假设 $ZZ^\top$ 满秩。 我们希望寻找在可以拟合数据的前提下泛化风险最小的估计：


$$
\begin{align*}
\min_{X\hat \beta = Y} R(\hat \beta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[(y - x^\top \hat \beta)^2 - (y - x^\top \beta )^2]
\end{align*}
$$


化简并且利用伪逆的性质可以得到 $\hat \beta$ 的显式解，


$$
\begin{align*}
&\quad \min_{X\hat \beta = Y} R(\hat \beta) = \mathbb{E}_{(x,y) \sim \mathcal{D}}[(y - x^\top \hat \beta)^2 - (y - x^\top \beta )^2]  \\
&= \min_{X\hat \beta = Y} \mathbb{E}_{(x,\epsilon) \sim \mathcal{D}}[(x^\top \beta + \epsilon - x^\top \hat \beta)^2 - \epsilon^2]  \\
&= \min_{X\hat \beta = Y}\mathbb{E}_{(x,\epsilon) \sim \mathcal{D}}[(x^\top \beta  - x^\top \hat \beta)^2 ]  \\
&= \min_{X\hat \beta = Y}(\hat \beta  - \beta)^\top \mathbb{E}_{(x,\epsilon) \sim \mathcal{D}}[xx^\top ] (\hat \beta  - \beta) \\
&= \min_{X\hat \beta = X \beta +W}(\hat \beta  - \beta)^\top \Sigma (\hat \beta  - \beta) \\
&=  \min_{X(\hat \beta - \beta) = W} \Vert \Sigma^{1/2} (\hat \beta  - \beta) \Vert_2^2 \\
&=  \min_{Z \Sigma^{1/2}(\hat \beta - \beta) = W} \Vert \Sigma^{1/2} (\hat \beta  - \beta) \Vert_2 \\
&= \min_{Z \alpha = W} \Vert \alpha \Vert_2^2 \\
&=  \Vert Z^\top (ZZ^\top)^{-1}W \Vert_2^2 \\
&= W^\top (ZZ^\top)^{-1} W.
\end{align*}
$$


直观地，当 $ p \gg n$ 时， 矩阵 $Z \in \mathbb{R}^{n \times p}$  的奇异值满足 : $\sigma (Z) \approx \sqrt{p}$.

而 $W^\top W$ 服从卡方分布，其值集中在均值附近，也即 $\mathbb{E}[W^\top W] = n \sigma^2$.

因此对应的理想插值器（Ideal Interpolator) 的泛化风险应该集中在 $R(\hat \beta) \approx n \sigma^2/p$. 

下面利用集中不等式给出上述描述的非渐进结果。



对于奇异值，采用随机高斯矩阵奇异值的双侧界，


$$
\begin{align*}
\sqrt{p} -  \sqrt n - 2\sqrt{\ln \frac{1}{\delta}}  \le \sigma(Z) \le \sqrt{p} + \sqrt{n} + 2\sqrt{\ln \frac{1}{\delta}} ,\quad {\rm with. prob. } 1- 2\delta.
\end{align*}
$$


对于 $W^\top W$ 的值，采用卡方分布的尾估计


$$
\begin{align*}
n \sigma^2 - \sqrt{8n \ln \frac{1}{\delta}} \le  W^\top W \le n \sigma^2 - \sqrt{8n \ln \frac{1}{\delta}}, \quad {\rm with.prob.} 1- 2\delta
\end{align*}
$$


代入可以得到 $R(\hat \beta) = \Theta(n \sigma^2/p)$ 的高概率界，当 $p$ 很大的时候，泛化风险将趋于零。



至此我们证明了对于理想插值器来说，的确存在无害过拟合的现象。但是至此我们仍然不知道如何寻找这样一个理想插值器，进一步的讨论请移步至原文章。





