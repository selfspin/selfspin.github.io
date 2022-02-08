---
title: 'VR-EXTRA and VR-DIGing'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 随机优化
  - 分布式优化
---



论文阅读笔记：[Variance Reduced EXTRA and DIGing and Their Optimal Acceleration for Strongly Convex Decentralized Optimization](https://arxiv.org/abs/2009.04373)



<!--more-->



本文利用方差缩减技术，结合Katyusha动量的加速方法等技术，将分布式随机优化问题同时做到了最优的计算复杂度和通讯复杂度。



## Variance Reduction



对于去中心化分布式问题，经典的手段是将其重新表述为，


$$
\min \sum_{i=1}^m f_i(x_i) ,\text{ s.t. } x_1 =x_2 = ... =x_m
$$


进一步重塑为，


$$
\begin{align}
\min_{\mathbf{x}} f(\mathbf{x}) + \frac{1}{2\alpha} \Vert V \mathbf{x} \Vert^2 ,\text{ s.t. } U \mathbf{x} = 0 \\
\end{align}
$$


其中对称正定矩阵 $U,V$ 满足，


$$
\begin{align}
Ux = 0 \text{ Iff } x_1 =x_2 = ... =x_m \\
Vx = 0 \text{ Iff } x_1 =x_2 = ... =x_m \\
\end{align}
$$


对于Lagrange函数使用梯度下降方法可以得到，


$$
\begin{align}
L(\mathbf{x}, \lambda) &= f(\mathbf{x}) + \frac{1}{2 \alpha} \Vert V \mathbf{x} \Vert^2 + \frac{1}{\alpha} \langle U \mathbf{x}, \lambda \rangle \\
\mathbf{x}_{k+1} &= \mathbf{x}_k - (\alpha \nabla f(\mathbf{x}) + U \lambda_k + V^2 \mathbf{x}_k) \\
\lambda_{k+1} &= \lambda_k + U\mathbf{x}_{k+1}
\end{align}
$$
结合方差缩减技术，每次采样大小为 $b$ 的MiniBatch，



$$
\begin{align}
\mathbf{s}_k &= \mathbb{E} [\nabla f(\mathbf{x_k},\xi_k) - \nabla f(\mathbf{w}_k,\xi_k)  + \nabla f(\mathbf{w_k}) ] \\
\mathbf{x}_{k+1} &= \mathbf{x}_k - (\alpha \mathbf{s}_k  + U \lambda_k + V^2 \mathbf{x}_k) \\
\lambda_{k+1} &= \lambda_k + U\mathbf{x}_{k+1} \\
\mathbb{w}_{k+1} &= \mathbf{x}_{k+1} ,\text{ with prob. } \frac{b}{n} \\
&= \mathbf{w}_k ,\text{ with prob. } \frac{b}{n}
\end{align}
$$



对于EXTRA或者DIGing算法，其定义的矩阵 $U,V$ 满足如下的重要性质，


$$
\begin{align}
\Vert U \mathbf{x} \Vert^2 &\le \Vert V \mathbf{x} \Vert^2 \le \frac{1}{2} \Vert \mathbf{x} \Vert^2 \\
\Vert U \lambda \Vert^2 &\ge \frac{1}{\kappa} \Vert \lambda \Vert^2, \forall  \lambda \in \text{span}(U)
\end{align}
$$


其中 $\kappa$ 为对应的条件数，对于 EXTRA算法， $\kappa= 2\kappa_c$ 而对于DIGing算法， $\kappa = \kappa_c^2$, 其中 $\kappa_c = \frac{1}{1 - \lambda_2(W)}$

上述性质仅当Gossip 矩阵 $W$ 满足某些条件的时候才成立，但可以使用Chebyshev 加速算法对矩阵 $W$ 进行 $\mathcal{O}(\sqrt{\kappa_c})$ 时间内的预处理使得上述条件满足。



对于 $L$-光滑的凸函数，梯度的差距存在如下重要引理，对于向量版本也成立，


$$
\begin{align}
\Vert \nabla f(\mathbf{y}) - \nabla f(\mathbf{x}) \Vert^2 \le 2L(f(\mathbf{y})- f(\mathbf{x}) - \langle \nabla f(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle)
\end{align}
$$




定义 $ x_{\ast} $ 为问题的解，而 $ \mathbf{x}_{\ast} = \mathbf{1} \cdot x_{\ast} $ 满足最优性条件，


$$
\begin{align}
\nabla f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} U \lambda^{\ast} = 0
\end{align}
$$


方差缩减技术本质上控制住了梯度的方差，而MiniBatch起到了减小方差的作用，



$$
\begin{align}
\mathbb{E} [\Vert \mathbf{s}_k - \nabla f(\mathbf{x}_k) \Vert^2 ] &= \mathbb{E} [ \Vert \nabla f(\mathbf{x}_k,\xi) - \nabla f(\mathbf{w}_k,\xi)  + \nabla f(\mathbf{w}_k) - \nabla f(\mathbf{x_k}) \Vert^2] \\
&\le \frac{1}{b}\mathbb{E} [ \Vert \nabla f(\mathbf{x}_k,\xi) - \nabla f(\mathbf{w}_k,\xi)  \Vert^2] \\
&\le \frac{2}{b} \mathbb{E} [\Vert \nabla f(\mathbf{x}_k, \xi) -  \nabla  f( \mathbf{x_{\ast}},\xi) \Vert^2] + \frac{2}{b} \mathbb{E} [ \Vert \nabla f(\mathbf{w}_k,\xi) - \nabla f(\mathbf{x_{\ast}},\xi) \Vert^2 ] \\
&\le \frac{4 L}{b} \mathbb{E} [f(\mathbf{x_k,\xi} ) -  f(\mathbf{x_{\ast}},\xi) - \langle \nabla f(\mathbf{x}_{\ast},\xi), \mathbf{x}_k - \mathbf{x}_{\ast} \rangle] + \frac{4 L}{b} \mathbb{E} [f(\mathbf{w_k} ,\xi) -  f(\mathbf{x_{\ast}},\xi) - \langle \nabla f(\mathbf{x}_{\ast},\xi), \mathbf{w}_k - \mathbf{x}_{\ast} \rangle] \\
&= \frac{4L}{b} (f(\mathbf{x}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle U \lambda_{\ast}, \mathbf{x}_k - \mathbf{x}_{\ast} \rangle)  +\frac{4L}{b} (f(\mathbf{w}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle U \lambda_{\ast},\mathbf{w}_k - \mathbf{x}_{\ast} \rangle) \\
&= \frac{4L}{b} (f(\mathbf{x}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle\lambda_{\ast}, U \mathbf{x}_k  \rangle)  +\frac{4L}{b} (f(\mathbf{w}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U\mathbf{w}_k  \rangle) \\
\end{align}
$$



进而，


$$
\begin{align}
f(\mathbf{x}_{k+1}) & \le f(\mathbf{x}_k) + \langle \nabla f(\mathbf{x}_k), \mathbf{x}_{k+1} - \mathbf{x}_k \rangle + \frac{L}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 \\
&= f(\mathbf{x}_k) + \langle \nabla  f(\mathbf{x}_k), \mathbf{x}_{k+1} - \mathbf{x}_k \rangle + \frac{L}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 \\
&= f(\mathbf{x}_k) + \langle \nabla  f(\mathbf{x}_k)- \mathbf{s}_k, \mathbf{x}_{k+1} - \mathbf{x}_k \rangle + \frac{L}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 + \langle \mathbf{s}_k,\mathbf{x}_{k+1} - \mathbf{x}_k \rangle \\
&\le f(\mathbf{x}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{x}_k) - \mathbf{s}_k \Vert^2 + \frac{L+\tau}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 + \langle \mathbf{s}_k,\mathbf{x}_{k+1} - \mathbf{x}_k \rangle \\
&= f(\mathbf{x}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{x}_k) - \mathbf{s}_k \Vert^2 + \frac{L+\tau}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 + \langle \mathbf{s}_k,\mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle + \langle \mathbf{s}_k,\mathbf{x}_{k} - \mathbf{x}_{\ast} \rangle\\
\end{align}
$$


继续Bound，回顾更新公式，


$$
\begin{align}
\mathbf{x}_{k+1} &= \mathbf{x}_k - (\alpha \mathbf{s}_k  + U \lambda_k + V^2 \mathbf{x}_k) \\ 
\mathbf{s}_k &= \frac{1}{\alpha} (\mathbf{x}_k - \mathbf{x}_{k+1} - U \lambda _k - V^2 \mathbf{x_k})
\end{align}
$$
因此，


$$
\begin{align}
\langle \mathbf{s}_k,\mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle &= \frac{1}{\alpha} \langle \mathbf{x}_k - \mathbf{x}_{k+1}, \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle - \frac{1}{\alpha} \langle \lambda_k, U\mathbf{x}_{k+1} \rangle - \frac{1}{\alpha} \langle V \mathbf{x}_k , V \mathbf{x}_{k+1} \rangle \\
&= \frac{1}{\alpha} \langle \mathbf{x}_k - \mathbf{x}_{k+1}, \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle - \frac{1}{\alpha} \langle \lambda_k, \lambda_{k+1} - \lambda_k \rangle - \frac{1}{\alpha} \langle V \mathbf{x}_k , V \mathbf{x}_{k+1} \rangle \\
&= \frac{1}{\alpha} \langle \mathbf{x}_k - \mathbf{x}_{k+1}, \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle + \frac{1}{\alpha} \langle \lambda_{\ast} - \lambda_k, \lambda_{k} - \lambda_{k+1} \rangle - \frac{1}{\alpha} \langle V \mathbf{x}_k , V \mathbf{x}_{k+1} \rangle  - \frac{1}{\alpha} \langle \lambda_{\ast}, \lambda_{k+1} - \lambda_k \rangle \\
&= \frac{1}{2\alpha} (\Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{x}_{k+1} - \mathbf{x}_{k} \Vert^2  - \Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2) - \frac{1}{2\alpha} ( \Vert \lambda_{\ast}  - \lambda_{k+1} \Vert^2 - \Vert \lambda_{k} - \lambda_{\ast} \Vert^2 - \Vert \lambda_{k+1} - \lambda_{k} \Vert^2) \\
&\quad - \frac{1}{2 \alpha}(\Vert V\mathbf{x}_k \Vert^2 + \Vert V\mathbf{x}_{k+1} \Vert^2 - \Vert V \mathbf{x}_k  - V \mathbf{x}_{k+1} \Vert^2 ) -\frac{1}{\alpha} \langle \lambda_{\ast} , U \mathbf{x}_{k+1} \rangle \\
&\le \frac{1}{2\alpha} (\Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{x}_{k+1} - \mathbf{x}_{k} \Vert^2  - \Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2) + \frac{1}{2\alpha} ( \Vert \lambda_{k}  - \lambda_{\ast} \Vert^2  - \Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2) \\
&\quad -\frac{1}{2 \alpha}(\Vert V\mathbf{x}_k \Vert^2  - \frac{1}{2}\Vert  \mathbf{x}_k  - \mathbf{x}_{k+1} \Vert^2 ) -\frac{1}{\alpha} \langle \lambda_{\ast} , U \mathbf{x}_{k+1} \rangle \\
&= \frac{1}{2 \alpha} ( \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2  - \Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2) + \frac{1}{2\alpha} ( \Vert \lambda_{k}  - \lambda_{\ast} \Vert^2  - \Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2) \\ 
&\quad -\frac{1}{2 \alpha}(\Vert V\mathbf{x}_k \Vert^2  - \frac{1}{2}\Vert  \mathbf{x}_k  - \mathbf{x}_{k+1} \Vert^2 ) -\frac{1}{\alpha} \langle \lambda_{\ast} , U \mathbf{x}_{k+1} \rangle \\
\end{align}
$$



因此，一方面，对不等式取条件期望，


$$
\begin{align}
\mathbb{E}[f(\mathbf{x}_{k+1})]& \le \mathbb{E} [f(\mathbf{x}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{x}_k) - \mathbf{s}_k \Vert^2 + \frac{L+\tau}{2} \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 + \langle \mathbf{s}_k,\mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle + \langle \mathbf{s}_k,\mathbf{x}_{k} - \mathbf{x}_{\ast} \rangle]\\
&\le f(\mathbf{x}_{\ast}) + \frac{1}{2 \tau} \mathbb{E} [\Vert \nabla f(\mathbf{x}_k) - \mathbf{s}_k \Vert^2] + \frac{L+\tau}{2} \mathbb{E}[ \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 ] - \frac{\mu}{2} \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 \\
&\quad + \frac{1}{2 \alpha} ( \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2  - \mathbb{E} [\Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) + \frac{1}{2\alpha} ( \Vert \lambda_{k}  - \lambda_{\ast} \Vert^2  - \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2]) \\
&\quad-\frac{1}{2 \alpha}(\Vert V\mathbf{x}_k \Vert^2  - \frac{1}{2}\mathbb{E}[\Vert  \mathbf{x}_k  - \mathbf{x}_{k+1} \Vert^2] ) -\frac{1}{\alpha} \mathbb{E} \langle \lambda_{\ast} , U \mathbf{x}_{k+1} \rangle 
\end{align}
$$


也即，


$$
\begin{align}
\mathbb{E}[ f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k+1} \rangle] &\le   \frac{1}{2 \tau} \mathbb{E} [\Vert \nabla f(\mathbf{x}_k) - \mathbf{s}_k \Vert^2] + \frac{L+\tau}{2} \mathbb{E}[ \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 ] - \frac{\mu}{2} \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 \\
&\quad + \frac{1}{2 \alpha} ( \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2  - \mathbb{E} [\Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) + \frac{1}{2\alpha} ( \Vert \lambda_{k}  - \lambda_{\ast} \Vert^2  - \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2]) \\
&\quad-\frac{1}{2 \alpha}(\Vert V\mathbf{x}_k \Vert^2  - \frac{1}{2}\mathbb{E}[\Vert  \mathbf{x}_k  - \mathbf{x}_{k+1} \Vert^2] ) 
\end{align}
$$


另一方面，


$$
\begin{align}
f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k+1} \rangle &= f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) - \langle \nabla f(\mathbf{x}_{\ast}), \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle \\
&\ge \frac{1}{2L} \Vert  \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_{\ast}) \Vert^2 \\
&= \frac{1}{2 L \alpha^2} \Vert \alpha \nabla f(\mathbf{x}_{k+1})+ U \lambda_{\ast}) \Vert^2 \\
&= \frac{1}{2 L \alpha^2} \Vert \mathbf{x}_k - \mathbf{x}_{k+1} - U \lambda _k - V^2 \mathbf{x_k} - \mathbf{s_k} + \nabla f(\mathbf{x}_{k+1}) + U \lambda_{\ast} \Vert^2 \\
&\ge  \frac{1- \nu}{2 L \alpha^2} \Vert U \lambda_k - U \lambda_{\ast} \Vert^2\\
&\quad - \frac{1}{2L\alpha^2}(\frac{1}{\nu}-1) \Vert  \mathbf{x}_k - \mathbf{x}_{k+1}  - V^2 \mathbf{x_k} - \mathbf{s_k} + \nabla f(\mathbf{x}_k ) - \nabla f(\mathbf{x}_k ) +\nabla f(\mathbf{x}_{k+1})  \Vert^2 \\
&\ge \frac{1- \nu}{2 \kappa L \alpha^2} \Vert  \lambda_k -  \lambda_{\ast} \Vert^2 -  \frac{2}{L\alpha^2} (1+ \alpha^2 L^2)(\frac{1}{\nu}-1) \Vert \mathbf{x}_{k+1}- \mathbf{x}_k \Vert^2\\
&\quad   - \frac{2}{L\alpha^2}(\frac{1}{\nu}-1) \Vert V \mathbf{x}_k \Vert^2 - \frac{2}{L\alpha^2}(\frac{1}{\nu}-1) \Vert \mathbf{s}_k - \nabla f(\mathbf{x}_k) \Vert^2
\end{align}
$$

取期望并且取上式的 $\frac{1}{2}$ 与之前的不等式进行组合，



$$
\begin{align}
\frac{1}{2}\mathbb{E}[ f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k+1} \rangle] &\le (\frac{1}{2 \alpha} - \frac{\mu}{2}) \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 - \frac{1}{2 \alpha} \mathbb{E} [\Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2] \\
&\quad + (\frac{1}{2 \alpha} - \frac{1 -\nu}{4\kappa L \alpha^2}) \Vert \lambda_k - \lambda_{\ast} \Vert^2 - \frac{1}{2 \alpha} \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2] \\
&\quad +(\frac{1}{2 \tau} + \frac{1}{L \alpha^2} (\frac{1}{\nu}-1)) \mathbb{E} [\Vert \mathbf{s}_k - \nabla f(\mathbf{x}_k) \Vert^2] \\
&\quad+  (\frac{L+\tau}{2}+ \frac{1}{L\alpha^2} (1+ \alpha^2L^2) (\frac{1}{\nu}-1) - \frac{1}{4 \alpha}) \mathbb{E}[ \Vert \mathbf{x}_{k+1} - \mathbf{x}_k \Vert^2 ] \\
&\quad + (\frac{1}{L\alpha^2} ( \frac{1}{\nu} - 1)-\frac{1}{2 \alpha}) \Vert V \mathbf{x}_k \Vert^2 \\
\end{align}
$$



选择合适的 $\tau, \nu, \alpha$, 可以满足，


$$
\begin{align}
\frac{1}{2 \tau} + \frac{1}{L \alpha^2} (\frac{1}{\nu}-1)) \le \frac{1}{24L} \\
\frac{L+\tau}{2}+ \frac{1}{L\alpha^2} (1+ \alpha^2L^2) (\frac{1}{\nu}-1) - \frac{1}{4 \alpha}) &\le 1 \\
\frac{1}{L\alpha^2} ( \frac{1}{\nu} - 1)-\frac{1}{2 \alpha} &\le 1\\
\end{align}
$$


因而，


$$
\begin{align}
\frac{1}{2}\mathbb{E}[ f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k+1} \rangle] &\le (\frac{1}{2 \alpha} - \frac{\mu}{2}) \Vert \mathbf{x}_k - \mathbf{x}_{\ast} \Vert^2 - \frac{1}{2 \alpha} \mathbb{E} [\Vert \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \Vert^2] \\
&\quad + (\frac{1}{2 \alpha} - \frac{1 -\nu}{4\kappa L \alpha^2}) \Vert \lambda_k - \lambda_{\ast} \Vert^2 - \frac{1}{2 \alpha} \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2] \\
&\quad + \frac{1}{6b} (f(\mathbf{x}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle\lambda_{\ast}, U \mathbf{x}_k  \rangle)  +\frac{1}{6b} (f(\mathbf{w}_k ) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U\mathbf{w}_k  \rangle) \\
\end{align}
$$


根据更新公式可以知道，


$$
\begin{align}
\mathbb{E}[ f(\mathbf{w}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{w}_{k+1} \rangle] &=  \frac{b}{n} \mathbb{E}[ f(\mathbf{x}_{k}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k} \rangle] +(1- \frac{b}{n}) \mathbb{E}[ f(\mathbf{w}_{k}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{w}_{k} \rangle]
\end{align}
$$


因此，定义，


$$
\begin{align}
\mathcal{D}_k &= \frac{1}{2} [ f(\mathbf{x}_{k}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k}] \\
\mathcal{W}_k &= \frac{\lambda}{2} [ f(\mathbf{w}_{k}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{w}_{k}] \\
\mathcal{X}_k &= \frac{1}{2 \alpha} \Vert \mathbf{x}_{k} - \mathbf{x}_{\ast} \Vert^2 \\
\mathcal{\Lambda}_k &= \frac{1}{2 \alpha} \Vert \lambda_{k} - \lambda_{\ast} \Vert^2 \\ 
\mathcal{V}_k &= \mathcal{D}_k +\mathcal{W}_k + \mathcal{X}_k + \mathcal{\Lambda}_k
\end{align}
$$



可以得到线性收敛的结果，



$$
\begin{align}
\mathbb{E} [\mathcal{V}_{k+1}] &\le (\frac{1}{3b}+ \frac{\lambda b}{n}) \mathcal{D}_k + (\frac{1}{3\lambda b}+ (1- \frac{b}{n})) \mathcal{W}_k +  (1- \mu \alpha) \mathcal{X}_k + (1-\frac{1-\nu}{2 \kappa L \alpha}) \mathcal{\Lambda}_k \\
&\le (1- \frac{b}{2n}) \mathcal{D_k} + (1 - \frac{b}{2n}) \mathcal{W}_k +  (1- \mu \alpha) \mathcal{X}_k+ (1-\frac{1-\nu}{2 \kappa L \alpha}) \mathcal{\Lambda}_k \\
&\le \max(1- \frac{b}{2n}, 1 - \mu \alpha, 1 - \frac{1-\nu}{2 \kappa L \alpha}) \mathcal{V}_k
\end{align}
$$





因此计算复杂度 $T$ 和通讯复杂度 $C$ 分别为，由于MiniBatch需要计算 $b$ 次梯度，因此计算复杂度为通讯复杂度的 $b$ 倍，


$$
\begin{align}
C &= \mathcal{O}((\frac{n}{b}+ \frac{L}{\mu} + \kappa) \log \frac{1}{\epsilon} ) \\
T &= \mathcal{O}((n + \frac{bL}{\mu} + b\kappa ) \log \frac{1}{\epsilon} )\\ 
\end{align}
$$

根据不同的情况，选取合适的  $b = \frac{n}{\kappa}$ ，注意到  $b<1$  的时候无意义，但可以引入一些零样本，使得等价于 $b=1$ 的情况，

使得复杂度为，

$$
\begin{align}
C &= \mathcal{O}((\kappa +\frac{L}{\mu} ) \log \frac{1}{\epsilon} ) \\
T &= \mathcal{O}((n + \frac{L}{\mu}  ) \log \frac{1}{\epsilon} )\\ 
\end{align}
$$



## Accelerated Algorithm



算法结合了Katyusha动量的公式，


$$
\begin{align}
\mathbf{y}_k &= \theta_1 \mathbf{z}_k + \theta_2 \mathbf{w}_k + (1- \theta_1 - \theta_2) \mathbf{x}_k \\
\mathbf{s}_k &= \mathbb{E}[\nabla f(\mathbf{y}_k,\xi) - \nabla f(\mathbf{w}_k,\xi)+ \nabla f(\mathbf{w}_k)] \\
\mathbf{z}_{k+1} &= \frac{1}{1 + \frac{\mu \alpha}{\theta_1}} ( \frac{\mu \alpha}{\theta_1 } \mathbf{y}_k + \mathbf{z}_k - \frac{1}{\theta_1} (\alpha \mathbf{s}_k +U \lambda_k + \theta_1 V^2 \mathbf{z}_k)) \\
\lambda_{k+1} &= \lambda_k + \theta_1 U \mathbf{z}_k \\
\mathbf{x}_{k+1} &= \mathbf{y}_k + \theta_1 (\mathbf{z}_{k+1} - \mathbf{z}_k)  \\
\mathbf{w}_{k+1} &= \mathbf{x}_{k} \text{ with prob.} \frac{b}{n} \\
&= \mathbf{w}_k \text{ with  prob. } 1-\frac{b}{n}
\end{align}
$$


下面基于Katyusha和VR-EXTRA and DIGing 的证明思路进行，


$$
\begin{align}
f(\mathbf{x}_{k+1}) &\le f(\mathbf{y}_k) + \langle f(\mathbf{y}_k). \mathbf{x}_{k+1} - \mathbf{y}_k \rangle + \frac{L}{2} \Vert \mathbf{x}_{k+1} - \mathbf{y}_k \Vert^2 \\
&\le f(\mathbf{y}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2 + \frac{L + \tau}{2} \Vert \mathbf{x}_{k+1} - \mathbf{y}_k \Vert^2  + \langle \mathbf{s}_k , \mathbf{x}_{k+1} - \mathbf{y}_k \rangle\\
&=f(\mathbf{y}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2 + \frac{(L + \tau) \theta_1^2}{2} \Vert \mathbf{z}_{k+1} - \mathbf{z}_k \Vert^2  + \theta_1\langle \mathbf{s}_k , \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \rangle + \theta_1 \langle \mathbf{s}_k , \mathbf{x}_{\ast} - \mathbf{z}_k \rangle \\
\end{align}
$$


利用更新公式，


$$
\begin{align}
\mathbf{s}_k &= \frac{\theta_1}{\alpha} ( \mathbf{z}_k - \mathbf{z}_{k+1}) + \mu ( \mathbf{y}_k - \mathbf{z}_{k+1}) - \frac{1}{\alpha} U \lambda_k - \frac{\theta_1}{\alpha} V^2 \mathbf{z}_k 
\end{align}
$$


因此，


$$
\begin{align}
\theta_1 \langle \mathbf{s}_k , \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \rangle &= \frac{\theta_1^2}{\alpha} \langle \mathbf{z}_k - \mathbf{z}_{k+1}, \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \rangle + \mu \theta_1 \langle \mathbf{y}_k - \mathbf{z}_{k+1} , \mathbf{z}_{k+1} - \mathbf{x}_{\ast}\rangle - \frac{\theta_1}{\alpha} \langle \lambda_k, U \mathbf{z}_{k+1} \rangle  - \frac{\theta_1^2}{ \alpha} \langle V \mathbf{z}_k , V \mathbf{z}_{k+1} \rangle \\
&=\frac{\theta_1^2}{\alpha} \langle \mathbf{z}_k - \mathbf{z}_{k+1}, \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \rangle + \mu \theta_1 \langle \mathbf{y}_k - \mathbf{z}_{k+1} , \mathbf{z}_{k+1} - \mathbf{x}_{\ast}\rangle - \frac{\theta_1}{\alpha} \langle \lambda_{\ast}, U \mathbf{z}_{k+1} \rangle  - \frac{\theta_1^2}{ \alpha} \langle V \mathbf{z}_k , V \mathbf{z}_{k+1} \rangle\\
&\quad - \frac{\theta_1}{\alpha} \langle \lambda_{\ast} - \lambda_k , \lambda_k - \lambda_{k+1} \rangle \\
&= \frac{\theta_1^2}{2\alpha} ( \Vert \mathbf{z}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{z}_k  - \mathbf{z}_{k+1} \Vert^2 - \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2) - \frac{1}{2\alpha} ( \Vert \lambda_{k+1} -\lambda_{\ast} \Vert^2 - \Vert \lambda_{\ast} - \lambda_k \Vert^2  -\Vert \lambda_{k+1} - \lambda_k \Vert^2) \\
&\quad + \frac{\mu \theta_1}{2} ( \Vert \mathbf{y}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{y_k} - \mathbf{z}_{k+1} \Vert^2 - \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2) + \frac{\theta_1^2}{2\alpha} ( \Vert V\mathbf{z}_{k+1}- V \mathbf{z}_k \Vert^2- \Vert V \mathbf{z}_k \Vert^2 - \Vert V \mathbf{z}_{k+1} \Vert^2) \\
&\quad - \frac{\theta_1}{\alpha} \langle \lambda_{\ast}, U \mathbf{z}_{k+1 \rangle} \\
&= \frac{\theta_1^2}{2\alpha} ( \Vert \mathbf{z}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2) - \frac{1}{2\alpha} ( \Vert \lambda_{k+1} -\lambda_{\ast} \Vert^2 - \Vert \lambda_{\ast} - \lambda_k \Vert^2)  - \frac{\theta_1}{\alpha} \langle \lambda_{\ast}, U \mathbf{z}_{k+1 \rangle} \\
&\quad  + \frac{\mu \theta_1}{2} ( \Vert \mathbf{y}_k - \mathbf{x}_{\ast} \Vert^2 - \Vert \mathbf{y_k} - \mathbf{z}_{k+1} \Vert^2 - \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2) - \frac{\theta_1^2}{2\alpha} \Vert V \mathbf{z}_k \Vert^2  - \frac{\theta_1^2}{4 \alpha} \Vert \mathbf{z}_{k+1} - \mathbf{z}_k \Vert^2
\end{align}
$$



以及，


$$
\begin{align}
\theta_1 \mathbb{E}[\langle \mathbf{s}_k, \mathbf{x}_{\ast} - \mathbf{z}_k \rangle] &=  \langle \nabla  f(\mathbf{y}_k) , \theta_1 \mathbf{x}_{\ast} + \theta_2 \mathbf{w}_k +(1 -\theta_1 - \theta_2) \mathbf{x}_k - \mathbf{y}_k \rangle  \\
&= \theta_1 \langle \nabla f(\mathbf{y}_k), \mathbf{x}_{\ast} - \mathbf{y}_k \rangle + \theta_2 \langle \nabla f(\mathbf{y}_k), \mathbf{w}_k - \mathbf{y}_k \rangle + (1-\theta_1 - \theta_2) \langle \nabla f(\mathbf{y_k}), \mathbf{x}_k - \mathbf{y}_k \rangle \\
&\le \theta_1 f(\mathbf{x}_{\ast}) - \theta_1 f(\mathbf{y}_k) + (1-\theta_1 - \theta_2) \langle \nabla f(\mathbf{y_k}), \mathbf{x}_k - \mathbf{y}_k \rangle- \frac{\mu \theta_1}{2} \Vert \mathbf{y}_k - \mathbf{x}_{\ast} \Vert^2 + \theta_2 \langle \nabla f(\mathbf{y}_k), \mathbf{w}_k - \mathbf{y}_k \rangle  \\
&\le \theta_1 f(\mathbf{x}_{\ast})+(1- \theta_1 - \theta_2) f(\mathbf{x}_k) - (1- \theta_2) f(\mathbf{y}_k) - \frac{\mu \theta_1}{2} \Vert \mathbf{y}_k - \mathbf{x}_{\ast} \Vert^2 + \theta_2 \langle \nabla f(\mathbf{y}_k), \mathbf{w}_k - \mathbf{y}_k \rangle
\end{align}
$$


合起来有，


$$
\begin{align}
\mathbb{E} [f(\mathbf{x}_{k+1})] &\le \mathbb{E} [f(\mathbf{y}_k) + \frac{1}{2 \tau} \Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2 + \frac{(L + \tau) \theta_1^2}{2} \Vert \mathbf{z}_{k+1} - \mathbf{z}_k \Vert^2  + \theta_1\langle \mathbf{s}_k , \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \rangle + \theta_1 \langle \mathbf{s}_k , \mathbf{x}_{\ast} - \mathbf{z}_k \rangle] \\
&\le \theta_1 f(\mathbf{x}_{\ast})+(1- \theta_1 - \theta_2) f(\mathbf{x}_k) +\theta_2  f(\mathbf{y}_k) \\
&\quad + \frac{1}{2 \tau} \mathbb{E}[\Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2] +  \frac{(L + \tau) \theta_1^2}{2} \mathbb{E} [\Vert \mathbf{z}_{k+1} - \mathbf{z}_k \Vert^2]  \\
&\quad +  \frac{\theta_1^2}{2\alpha} ( \Vert \mathbf{z}_k - \mathbf{x}_{\ast} \Vert^2 - \mathbb{E} [\Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) + \frac{1}{2\alpha} ( \Vert \lambda_{k} -\lambda_{\ast} \Vert^2 - \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2])  - \frac{\theta_1}{\alpha} \mathbb{E}[ \langle \lambda_{\ast}, U \mathbf{z}_{k+1 \rangle}] \\
&\quad  - \frac{\mu \theta_1}{2} ( \mathbb{E} [\Vert \mathbf{y_k} - \mathbf{z}_{k+1} \Vert^2] + \mathbb{E}[ \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) - \frac{\theta_1^2}{2\alpha} \Vert V \mathbf{z}_k \Vert^2  - \frac{\theta_1^2}{4 \alpha} \mathbb{E} [\Vert \mathbf{z}_{k+1} -  \mathbf{z}_k \Vert^2]  +\theta_2 \langle \nabla f(\mathbf{y}_k), \mathbf{w}_k - \mathbf{y}_k \rangle\\
&\le \theta_1 f(\mathbf{x}_{\ast})+(1- \theta_1 - \theta_2) f(\mathbf{x}_k) +\theta_2  f(\mathbf{w}_k) \\
&\quad  + \frac{1}{2 \tau} \mathbb{E}[\Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2] - \theta_2 (f(\mathbf{w}_k ) - f(\mathbf{y}_k ) - \langle \nabla f(\mathbf{y_k}), \mathbf{w}_k - \mathbf{y}_k \rangle) \\
&\quad +\frac{\theta_1^2}{2\alpha} ( \Vert \mathbf{z}_k - \mathbf{x}_{\ast} \Vert^2 - \mathbb{E} [\Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) + \frac{1}{2\alpha} ( \Vert \lambda_{k} -\lambda_{\ast} \Vert^2 - \mathbb{E} [\Vert \lambda_{k+1} - \lambda_{\ast} \Vert^2])  - \frac{\theta_1}{\alpha} \mathbb{E}[ \langle \lambda_{\ast}, U \mathbf{z}_{k+1 \rangle}] \\ 
&\quad  - \frac{\mu \theta_1}{2} ( \mathbb{E} [\Vert \mathbf{y_k} - \mathbf{z}_{k+1} \Vert^2] + \mathbb{E}[ \Vert \mathbf{z}_{k+1} - \mathbf{x}_{\ast} \Vert^2]) - \frac{\theta_1^2}{2\alpha} \Vert V \mathbf{z}_k \Vert^2  - \frac{\theta_1^2}{4 \alpha} \mathbb{E} [\Vert \mathbf{z}_{k+1} -  \mathbf{z}_k \Vert^2] \\
\end{align}
$$


类似地，利用更新公式，

$$
\begin{align}
\alpha \mathbf{s}_k &= \theta_1 ( \mathbf{z}_k - \mathbf{z}_{k+1}) + \mu \alpha ( \mathbf{y}_k - \mathbf{z}_{k+1}) - U \lambda_k - \theta_1 V^2 \mathbf{z}_k 
\end{align}
$$






因此


$$
\begin{align}
f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) + \frac{1}{\alpha} \langle \lambda_{\ast}, U \mathbf{x}_{k+1} \rangle &= f(\mathbf{x}_{k+1}) - f(\mathbf{x}_{\ast}) - \langle \nabla f(\mathbf{x}_{\ast}), \mathbf{x}_{k+1} - \mathbf{x}_{\ast} \rangle \\
&\ge \frac{1}{2L} \Vert  \nabla f(\mathbf{x}_{k+1}) - \nabla f(\mathbf{x}_{\ast}) \Vert^2 \\
&= \frac{1}{2 L \alpha^2} \Vert \alpha \nabla f(\mathbf{x}_{k+1})+ U \lambda_{\ast}) \Vert^2 \\
&= \frac{1}{2 L \alpha^2} \Vert \theta_1 ( \mathbf{z}_k - \mathbf{z}_{k+1}) + \mu \alpha ( \mathbf{y}_k - \mathbf{z}_{k+1}) - U (\lambda_k - \lambda_{\ast}) - \theta_1 V^2 \mathbf{z}_k  - \alpha \mathbf{s}_k + \alpha \nabla f(\mathbf{y}_{k}) - \alpha \nabla f(\mathbf{y_k}) +\alpha \nabla f(\mathbf{x}_{k+1}) \Vert^2 \\
&\ge \frac{1- \nu}{ 2 \kappa L \alpha^2} \Vert \lambda_k - \lambda_{\ast} \Vert^2 -  \frac{5 \mu^2}{2 L }  (\frac{1}{\nu}-1) \Vert \mathbf{y}_k - \mathbf{z}_{k+1} \Vert^2 - \frac{5\theta_1^2}{2 L\alpha^2}(\frac{1}{\nu}-1) \Vert V^2 \mathbf{z}_k \Vert^2 \\
&\quad +\frac{5 }{2 L} \Vert \nabla f(\mathbf{y}_k) - \mathbf{s}_k \Vert^2 - \frac{5 \mu^2}{2 L }  (\frac{1}{\nu}-1)(1+ L^2 \alpha^2) \Vert \mathbf{y}_k - \mathbf{z}_{k+1} \Vert^2 \\

\end{align}
$$


利用两个关系式 进行线性组合，再进行参数的选择，可以证明最终算法可以达到的收敛率为，


$$
\begin{align}
C &= \mathcal{O}(\sqrt{\kappa \frac{L}{\mu}} \log \frac{1}{\epsilon}) \\
T &= \mathcal{O}((n + \sqrt{n \frac{L}{\mu}}) \log \frac{1}{\epsilon})
\end{align}
$$
