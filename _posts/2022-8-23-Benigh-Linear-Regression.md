---
title: 'Benigh Overfitting in Linear Regression'
toc: true
excerpt_separator: <!--more-->
tags:
  - 线性回归
---



Paper Reading: Benigh Overfitting in Linear Regression.



<!--more-->



本文探究线性回归模型在何种条件下可以产生无害过拟合：在训练集上损失为0，但是测试集上泛化误差也可以趋于0的情况。



## Introduction

模型的标准的线性回归模型，



$$
\begin{align*}
y =  x^\top \beta + \epsilon, \quad x \in \mathbb{R}^{p \times 1}, \quad y,\epsilon \in \mathbb{R}.
\end{align*}
$$



定义 $\Sigma = \mathbb{E}[xx^\top]$, 假设在给定特征的条件下，噪声 $\epsilon \mid x$ 为 $\sigma$-次高斯随机变量，而标准化后的数据特征 $z = \Sigma^{-1/2}x$ 的每一个元素为独立地 $\sigma_x$-次高斯随机变量，并且在文章中将 $\sigma_x^2$ 视作为常数。





抽取数据集 $(X,Y)$ , 数据上的噪声 $W \mid X$ 的每一行为 $\sigma$ -次高斯随机变量。



$$
\begin{align*}
Y = X \beta +W, \quad X \in \mathbb{R}^{n \times p}, \quad Y, \epsilon \in \mathbb{R}^{n \times 1}.
\end{align*}
$$



当 $XX^\top$ 满秩的时候，最小范数最小二乘解将是一个插值器，也即其一定满足 $X \hat \beta = Y$: 



$$
\begin{align*}
\hat \beta = X^\top(XX^\top)^{-1} Y.
\end{align*}
$$



定义泛化误差，并且我们考虑其上界， 并用矩阵 $B,C$ 简化表达式，



$$
\begin{align*}
R(\hat \beta) &= \mathbb{E}_{(x,y)} [ (x^\top \hat \beta - y)^2 - (x^\top \beta - y)^2] \\
&=\mathbb{E}_{(x,\epsilon)} [ (x^\top \hat \beta - x^\top \beta +\epsilon)^2 - \epsilon^2] \\
&= \mathbb{E}_{x} [ (x^\top (\hat \beta  - \beta))^2] \\
&= \mathbb{E}_{x} [ (x^\top (\hat \beta  - \beta))^2] \\
&= \mathbb{E}_{x} [ (x^\top (X^\top (XX^\top)^{-1} Y  - \beta))^2] \\
&= \mathbb{E}_{x} [ (x^\top (X^\top (XX^\top)^{-1} (X \beta +W)  - \beta))^2] \\
&\le 2 \mathbb{E}_x[(x^\top (I - X^\top (XX^\top)^{-1}X )\beta)^2  ] + 2 \mathbb{E}_x [ (x^\top X^\top (XX^\top)^{-1} W)^2] \\
&= 2\beta^\top B \beta + 2W^\top C W.
\end{align*}
$$



上述两项分别对应于偏差项和方差项，其中



$$
\begin{align*}
B &=  (I - X^\top (XX^\top)^{-1}X) \Sigma (I - X^\top (XX^\top)^{-1}X)  \\
C &= (XX^\top)^{-1} X \Sigma X^\top (XX^\top)^{-1}.
\end{align*}
$$




## Bound for Bias



对于偏差项，采用高斯分布的样本协方差的估计的界，参见 [arXiv](https://arxiv.org/pdf/1405.2468.pdf), 成立



$$
\begin{align*}
\left\Vert \Sigma - \frac{1}{n} XX^\top\right\Vert \le C \Vert \Sigma \Vert  \max \left\{ \sqrt{\frac{r(\Sigma)}{n}}, \frac{r(\Sigma)}{n},\sqrt{\frac{\log \frac{1}{\delta}}{n}} , \frac{\log \frac{1}{\delta}}{n} \right\}, {\rm with. prob.} 1- \delta
\end{align*}
$$



因此，



$$
\begin{align*}
{\rm bias} &= \beta^\top B \beta \\
&=  \beta^\top (I - X^\top (XX^\top)^{-1}X) \Sigma (I - X^\top (XX^\top)^{-1}X) \beta \\
 &= \beta^\top (I - X^\top (XX^\top)^{-1}X) \left(\Sigma - \frac{1}{n} XX^\top\right) (I - X^\top (XX^\top)^{-1}X) \beta \\
 &\le \Vert \beta \Vert^2  \left\Vert I - X^\top (XX^\top)^{-1}X  \right \Vert^2 \left\Vert \Sigma - \frac{1}{n} XX^\top\right\Vert \\
 &\le C\Vert \beta \Vert^2  \Vert \Sigma \Vert \max \left\{ \sqrt{\frac{r(\Sigma)}{n}}, \frac{r(\Sigma)}{n},\sqrt{\frac{\log \frac{1}{\delta}}{n}} , \frac{\log \frac{1}{\delta}}{n} \right\}, {\rm with. prob.} 1- \delta
\end{align*}
$$



其中 $r(\Sigma)$ 为稳定维数， 定义为 $r(\Sigma) = {\rm tr}(\Sigma) / \Vert \Sigma \Vert$.

 当 $n \rightarrow \infty$ 的时候，上述的偏差项将趋于0，下面我们关注在 $n$ 很大的时候方差项上是否也将趋于0，这将是文章的重点。



## Bound for Variance



首先将方差项用 ${\rm tr}(C)$ 控制，使用的不等式可以看作Hanson-Wright的条件分布版本 (原论文 Lemma 19)



$$
\begin{align*}
{\rm variance} = W^\top C W  \le 2 \sigma^2 {\rm tr}(C) + 4  \sigma^2 {\rm tr}(C) \log \frac{1}{\delta}, \quad {\rm with. prob.} 1- \delta.
\end{align*}
$$


而对于 ${\rm tr}(C)$, 使用Sherman-Woodbury-Morrison公式，可以得到

$$
\begin{align*}
{\rm tr}(C) &= {\rm tr}((XX^\top)^{-1} X \Sigma X^\top (XX^\top)^{-1}) \\
&= {\rm tr}( X \Sigma X^\top (XX^\top)^{-2}) \\
&= {\rm tr}( Z \Sigma^2 Z^\top (Z\Sigma Z^\top)^{-2}) \\
&= {\rm tr}( Z \Sigma^2 Z^\top A^{-2}) \\
&=\sum_{i\le k} \lambda_i^2 z_i^\top  \left(\sum_j \lambda_j z_j z_j^\top \right)^{-2} z_i + \sum_{i > k } \lambda_i^2 z_i^\top A^{-2} z_i \\
&= \sum_{i \le k} \lambda_i^2 z_i^\top  \left(\lambda_i z_i z_i^\top + A_{-i} \right)^{-2} z_i + \sum_{i > k } \lambda_i^2 z_i^\top A^{-2} z_i \\
&= \sum_{i \le k} \frac{\lambda_i^2 z_i^\top A_{{-i}}^{-2} z_i}{ (1 + \lambda_i z_i^\top A_{-i}^{-1} z_i)^2} + \sum_{i > k } \lambda_i^2 z_i^\top A^{-2} z_i \\
&\le \sum_{i \le k} \frac{ z_i^\top A_{{-i}}^{-2} z_i}{ (  z_i^\top A_{-i}^{-1} z_i)^2} + \sum_{i > k } \lambda_i^2 z_i^\top A^{-2} z_i \\ 
& \le \sum_{i \le k} \frac{ z_i^\top A_{{-i}}^{-2} z_i}{ ( (\Pi_{k: \infty} z_i)^\top A_{-i}^{-1} (\Pi_{k:\infty}z_i))^2} + \sum_{i > k } \lambda_i^2 z_i^\top A^{-2} z_i\\
&\le \sum_{i \le k} \frac{\mu_n^{-2}(A_{-i}) \Vert z_i \Vert^2}{ \mu_{k+1}^{-2} (A_{-i}) \Vert \Pi_{k: \infty}z_i \Vert^4} + \sum_{i > k} \lambda_i^2 \mu_n^{-2} (A) \Vert z_i \Vert^2 \\
&= \sum_{i \le k} \frac{\mu_{k+1}^{2}(A_{-i}) \Vert z_i \Vert^2}{ \mu_{n}^{2} (A_{-i}) \Vert \Pi_{k: \infty}z_i \Vert^4} + \sum_{i > k} \frac{\lambda_i^2 \Vert z_i \Vert^2}{\mu_n^{2}(A)}.
\end{align*}
$$


其中，为了记号方便我们定义



$$
\begin{align*}
A = \sum_{i} \lambda_i z_i z_i^\top, \quad A_{-i} = \sum_{j \ne i} \lambda_j z_j z_j^\top, \quad A_k = \sum_{i>k} \lambda_i z_i z_i^\top
\end{align*}
$$



以及定义 $\Pi_{k : \infty}$ 表示向 $A_{-i}^{-1}$ 的后$k$ 个特征值的投影，这一步较为巧妙，其背后所蕴含的深意是我们将有


$$
\begin{align*}
\frac{1}{c}K(A_k) \le \mu_n(A_{-i}) , \quad \mu_{k+1}(A_{-i}) \le c K(A_k).
\end{align*}
$$
 

其中 $K(A_k)$ 表示某种与 矩阵 $A_k$ 相关的界，该界也将被文章定义为某种有效维数，我们将在后文详细说明。

下面利用 $\epsilon$- 网理论控制矩阵 $A \in \mathbb{R}^{n \times n}$ 的特征值，利用 $z_i$ 为$\sigma_x$-次高斯随机变量并且将 $\sigma_x$ 视为常数，


$$
\begin{align*}
& \quad \left \Vert A - I_n \sum_{i} \lambda_i \right \Vert \\
& \le (1 - \epsilon)^{-2}  \max_{v \in \mathcal{N}_{\epsilon}} \left \vert v^\top A v - \sum_{i} \lambda_i \right\vert \\
&= (1 - \epsilon)^{-2}  \max_{v \in \mathcal{N}_{\epsilon}} \left \vert \sum_i \lambda_i (z_i^\top v)^2 - \sum_{i} \lambda_i \right\vert \\
&\le c (1- \epsilon)^{-2}  \max\left\{ \lambda_1 \log \frac{1}{\delta}, \sqrt{\sum_i \lambda_i^2 \log \frac{1}{\delta}} \right\},  \quad \forall v \in \mathcal{N}_{\epsilon}, \quad {\rm with. prob.} 1-2\delta \\
&\le c  (1- \epsilon)^{-2}  \max\left\{ \lambda_1 \log \frac{1}{\delta}, \sqrt{\sum_i \lambda_i \log \frac{1}{\delta}} \right\},  \quad \forall v \in \mathcal{N}_{\epsilon}, \quad {\rm with.prob.} 1-2\delta.
\end{align*}
$$


取 $\epsilon = 1/4$, 并且对所有的 $v \in \mathcal{N}_{\epsilon}$ 取一致界，利用 $\vert \mathcal{N}_{\epsilon} \vert \le 9^n$ 可以得到，


$$
\begin{align*}
& \quad \left \Vert A - I_n \sum_{i} \lambda_i \right \Vert \\ 
&\le c \ \max\left\{ \lambda_1 \left( n + \log \frac{1}{\delta} \right) , \sqrt{\sum_i \lambda_i^2 \left( n + \log \frac{1}{\delta}\right) } \right\}, \quad {\rm with. prob.} 1- 2\delta \\
&\le c \lambda_1 \left( n + \log \frac{1}{\delta}\right) + \gamma \sum_{i} \lambda_i, \quad \forall \gamma \in(0,1), \quad {\rm with. prob.} 1- 2\delta.
\end{align*}
$$


据此可以得到矩阵 $A$ 的奇异值的双侧界，


$$
\begin{align*}
\frac{1}{c} \sum_i \lambda_i - c \lambda_1 \left(n + \frac{1}{\delta} \right) \le \mu_n(A) \le \mu_1(A) \le c \sum_i \lambda_i +  c \lambda_1 \left(n + \frac{1}{\delta} \right),\quad {\rm with. prob.} 1- 2\delta.
\end{align*}
$$


将上述关于奇异值的双侧界嵌入 $A_k$ 并且利用 $A_k, A_{-i}, A$ 特征值的关系，可以直接得到，


$$
\begin{align*}
\mu_{k+1}(A_{-i}) \le \mu_{k+1} (A) \le \mu_1(A_k) &\le c \sum_{j > k } \lambda_j + c \lambda_{k+1} \left( n + \frac{1}{\delta}\right) ,\quad \forall i \ge 1 ,\quad {\rm with. prob.} 1- 2\delta.\\
\mu_n(A) \ge \mu_n(A_{-i}) \ge \mu_n(A_k) &\ge \frac{1}{c} \sum_{j>k} \lambda_j - c \lambda_{k+1} \left( n + \frac{1}{\delta}\right), \quad i \le k, \quad {\rm with. prob.} 1- 2\delta..
\end{align*}
$$


定义一下两个有效秩 $r_k, R_k$ , 在此基础上可以定义有效维数 $k^*$： 


$$
\begin{align*}
r_k (\Sigma) = \frac{\sum_{i >k} \lambda_i}{\lambda_{k+1}}, \quad R_k(\Sigma) = \frac{\left( \sum_{i>k} \lambda_i\right)^2}{\sum_{i >k} \lambda_i^2}, \quad k^* = \min\{k: r_k(\Sigma) \ge b (n + \log (\delta^{-1}))\},
\end{align*}
$$


将有效维数代入关于方差项的界中可以得到，


$$
\begin{align*}
{\rm variance} &\le c \sigma^2 \left( 1+ \log \frac{1}{\delta}\right) {\rm tr}(C), \quad {\rm with. prob.} 1- \delta \\
&\le c \sigma^2 \left( 1+ \log \frac{1}{\delta}\right) \left(\sum_{i \le k^*} \frac{\mu_{k^*+1}^{2}(A_{-i}) \Vert z_i \Vert^2}{ \mu_{n}^{2} (A_{-i}) \Vert \Pi_{k^*: \infty}z_i \Vert^4} + \sum_{i > k^*} \frac{\lambda_i^2 \Vert z_i \Vert^2}{\mu_n^{2}(A)}\right), \quad {\rm with. prob.} 1- \delta \\
&\le c \sigma^2 \left( 1+ \log \frac{1}{\delta}\right) \left( \sum_{i \le k^*} \frac{\Vert z_i \Vert^2}{\Vert \Pi_{k^*:\infty}  z_i \Vert^4} +   \frac{ \sum_{i > k^*}\lambda_i^2 \Vert z_i \Vert^2}{\left(\sum_{i > k^*} \lambda_i\right)^2} \right) , \quad {\rm with. prob.} 1- 7 \delta. 
\end{align*}
$$


而由于 $z_i$ 为 $\sigma_x$-次高斯随机变量，利用集中不等式可以得到其界，首先其范数集中在 $\sqrt{n}$ 附近


$$
\begin{align*}
n - c  \max \left\{  \log \frac{1}{\delta} , \sqrt{n \log \frac{1}{\delta}}\right\} \le \Vert z_i \Vert^2\le n - c \max \left\{  \log \frac{1}{\delta} , \sqrt{n \log \frac{1}{\delta}}\right\} , \quad{\rm with.prob.} 1- 2\delta.
\end{align*}
$$


另一方面，利用之前推导的条件二次型的界，


$$
\begin{align*}
&\quad \Vert \Pi_{k^*: \infty} z_i \Vert^2 \\ &= \Vert z_i \Vert^2 - \Vert \Pi_{0:k^*} z_i \Vert^2 \\
&= \Vert z_i \Vert^2 - z_i^\top \Pi_{0:k^*} z_i \\
&\ge \Vert z_i \Vert^2 - c k^* \left( 1+ \log \frac{1}{\delta} \right), \quad {\rm with.prob.} 1- \delta  \\
&\ge n - c  \max \left\{  \log \frac{1}{\delta} , \sqrt{n \log \frac{1}{\delta}}\right\} - ck^* \left( 1 + \log \frac{1}{\delta}\right),  \quad {\rm with. prob.} 1-3 \delta.
\end{align*}
$$


将上述得到的界都嵌入方差项的估计可以当 $n \ge a k^* (1 + \log (\delta^{-1)})$ 的时候得到最终的结果满足，


$$
\begin{align*}
{\rm variance} &\le \mathcal{O} \left( \frac{\sigma^2 k^*}{n} + \frac{\sigma^2 n }{R_{k^*}(\Sigma)} \right)
\end{align*}
$$


如果数据的协方差满足以下的条件即可形成无害过拟合：


$$
\begin{align*}
\lim_{n \rightarrow \infty} \frac{r(\Sigma)}{n}  =\frac{k^*}{n} = \frac{n}{R_{k^*}(\Sigma)} = 0.
\end{align*}
$$


并且文章还给出了对应的下界，说明上界的紧度。



