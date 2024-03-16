---
title: 'Gradient Descent with Locally Lipschitz Gradients'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - 优化
---



Paper Reading: Adaptive Gradient Descent without Descent.



<!--more-->



常见的梯度下降算法假设函数梯度Lipschitz，也即



$$
\begin{align*}
\Vert \nabla f(x) - \nabla f(y) \Vert \le L \Vert x - y \Vert.
\end{align*}
$$



本文考虑上述性质只在局部成立的情形，也即函数仅Locally Lipschitz的条件，

此时我们不知道Lipschitz系数，希望采取如下的方式进行估计



$$
\begin{align*}
L_k = \frac{ \Vert \nabla f(x^k) - \nabla f(x^{k-1}) \Vert }{ \Vert x^k - x^{k-1} \Vert}.
\end{align*}
$$



并且令步长 $\lambda_k \le 1/(2L_k)$, 下面我们分析上述算法,

假设函数满足凸性，定义 $\theta_k = \lambda_k / \lambda_{k-1}$, 



$$
\begin{align*}
&\quad \Vert x^{k+1} - x^\ast \Vert^2 \\
&= \Vert x^k - x^\ast \Vert + 2 \langle x^{k+1} - x^k , x^k - x^\ast \rangle + \Vert x^{k+1} - x^k \Vert^2 \\
&=  \Vert x^k - x^\ast \Vert - 2 \lambda_k \langle \nabla f(x^k), x^k - x^\ast \rangle + \Vert x^{k+1} - x^k \Vert^2 \\
&\le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) + \Vert x^{k+1} - x^k \Vert^2 \\
&=  \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) + 2\Vert x^{k+1} - x^k \Vert^2 - \Vert x^{k+1} - x^k \Vert^2 \\
&\le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) \\
&\quad - 2 \lambda_k \langle \nabla f(x^k), x^{k+1} - x^k \rangle - \Vert x^{k+1} - x^k \Vert^2 \\
&= \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) \\
&\quad +2 \lambda_k \langle \nabla f(x^k) - \nabla f(x^{k-1}), x^k- x^{k+1} \rangle  + 2 \lambda_k \langle \nabla f(x^{k-1}), x^k - x^{k+1} \rangle - \Vert x^{k+1} - x^k \Vert^2 \\
&\le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) \\
&\quad +2 \lambda_k \Vert \nabla f(x^k) - \nabla f(x^{k-1}) \Vert \Vert  x^k- x^{k+1} \Vert  + 2 \lambda_k \langle \nabla f(x^{k-1}), x^k - x^{k+1} \rangle - \Vert x^{k+1} - x^k \Vert^2 \\
&\le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) \\
&\quad + \frac{1}{2} \Vert x^k - x^{k-1} \Vert^2 -\frac{1}{2} \Vert x^{k+1} - x^k \Vert^2     + 
\frac{2 \lambda_k}{\lambda_{k-1}} \langle x^{k-1} - x^k, x^k - x^{k+1} \rangle  \\
& \le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast)  \\
& \quad + \frac{1}{2} \Vert x^k - x^{k-1} \Vert^2 -\frac{1}{2} \Vert x^{k+1} - x^k \Vert^2 + 2 \lambda_k \theta_k \langle x^{k-1} - x^k, \nabla f(x^k) \rangle \\
&\le \Vert x^k - x^\ast \Vert^2 - 2 \lambda_k (f(x^k) - f^\ast) \\
&\quad +  \frac{1}{2} \Vert x^k - x^{k-1} \Vert^2 -\frac{1}{2} \Vert x^{k+1} - x^k \Vert^2 + 2 \lambda_k \theta_k (f(x^{k-1}) - f(x^k)).
\end{align*}
$$



整理后得到



$$
\begin{align*}
&\quad  \Vert x^{k+1} - x^\ast \Vert^2 + \frac{1}{2} \Vert x^{k+1} - x^k \Vert^2 + 2 \lambda_k (\theta_k +1) (f(x^k) - f^\ast) \\
&\le \Vert x^k - x^\ast \Vert^2 + \frac{1}{2} \Vert x^k - x^{k-1} \Vert^2 + 2 \lambda_k \theta_k (f(x^{k-1}) - f^\ast).
\end{align*}
$$



这说明迭代中的序列是有界的，具体地，应该满足



$$
\begin{align*}
\Vert x^k - x^\ast \Vert^2 \le \Vert x^1 - x^\ast \Vert^2+ \frac{1}{2} \Vert x^1 - x^0 \Vert^2 + 2 \lambda_1 \theta_1 (f(x^0) - f^\ast) := D.
\end{align*}
$$



定义



$$
\begin{align*}
w_i = \lambda_i (1 + \theta_i) - \lambda_{i+1} \theta_{i+1},
\end{align*}
$$



令步长 $\lambda_k$ 始终满足


$$
\begin{align*}
\lambda_k^2 \le \lambda_{k-1}^2 (1 + \theta_{k-1})
\end{align*}
$$


则权重均为正数，也即 $w_i \ge 0$. 从上面的式子也可以得到



$$
\begin{align*}
\lambda_k (\theta_{k}  + 1) (f(x^k) - f^\ast) + \sum_{i=1}^{k-1} w_i  (f(x^i) - f^\ast) \le \frac{D}{2}. 
\end{align*}
$$


输出


$$
\begin{align*}
\hat x  = \lambda_k (\theta_k +1) x_k + \sum_{i=1}^{k-1} w_{i} x_{k-1}.
\end{align*}
$$



由于序列始终在有界集中，存在常数 $L$ 为该集合中的梯度Lipschitz系数，根据步长的公式


$$
\begin{align*}
\lambda_{k} = \min \left \{ \sqrt{1+ \lambda_{k-1}/ \lambda_{k-2}} \lambda_{k-1}, \frac{\Vert x_k - x_{k-1} \Vert}{2 \Vert \nabla f(x_k) - \nabla f(x_{k-1}) \Vert }  \right \}.
\end{align*}
$$


可以根据归纳得到 $\lambda_k \ge 1/(2L)$, 这意味着


$$
\begin{align*}
S_K :=  \lambda_k (\theta_k +1)  + \sum_{i=1}^{k-1} w_{i}=  \sum_{i=1}^k \lambda_i + \lambda_1 \theta_1 \ge \frac{k}{2L}. 
\end{align*}
$$


那么根据函数的凸性以及Jensen‘s 不等式，得到


$$
\begin{align*}
f(\hat x) - f^\ast \le \frac{L D}{k}.
\end{align*}
$$


这与已经全局梯度Lipschitz系数下选取步长为 $\lambda_k = 1/(2L)$ 时的收敛率相当。

