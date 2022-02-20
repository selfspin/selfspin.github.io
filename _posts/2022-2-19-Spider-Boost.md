---
title: 'SpiderBoost'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 随机优化
  - 非凸优化
---



论文阅读笔记： [Spiderboost and momentum: Faster variance reduction algorithms](https://proceedings.neurips.cc/paper/2019/hash/512c5cad6c37edb98ae91c8a76c3a291-Abstract.html)

<!--more-->



## Offline Setting

算法每隔 $q$ 轮计算全梯度 $v_k = \nabla f(x_k)$  ,其他时候使用如下式子计算梯度 

$$
\begin{align*}
v_{k}  &= \frac{1}{\vert S \vert} \sum_{i \in S} \nabla f_i(x_k)- \nabla f_i(x_{k-1} ) +v_{k-1} 
\end{align*}
$$

如此进行方差缩减，与SVRG中的无偏估计不同，Spider采用的是有偏估计，但利用鞅的性质可以缩小其方差，


$$
\begin{align*}
\mathbb{E}[ \Vert v_k - \nabla f(x_k) \Vert^2] &\le \frac{L^2}{\vert S \vert} \mathbb{E} [ \Vert x_k - x_{k-1} \Vert^2 ] + \mathbb{E}[\Vert v_{k-1} - \nabla f(x_{k-1}) \Vert^2] \\
&= \frac{L^2 \eta_k^2}{\vert S \vert} \mathbb{E} [ \Vert v_k \Vert^2] + \mathbb{E}[\Vert v_{k-1} - \nabla f(x_{k-1}) \Vert^2] \\
\end{align*}
$$


递推可以得到方差的界，


$$
\begin{align*}
\mathbb{E}[ \Vert v_k - \nabla f(x_k) \Vert^2] &\le \frac{L^2 \eta_k^2}{\vert S \vert}\sum_{i=(n_k - 1)q}^k \mathbb{E}[\Vert v_i \Vert^2] + \sigma^2
\end{align*}
$$


利用 $L$ - 光滑的性质，


$$
\begin{align*}
\mathbb{E} [f(x_{k+1})] &\le \mathbb{E} [f(x_k) + \nabla f(x_k)^\top(x_{k+1} - x_k)  + \frac{L}{2} \Vert x_{k+1} - x_k \Vert^2] \\
&= \mathbb{E} [f(x_k) - \eta_k \nabla f(x_k)^\top v_k + \frac{L \eta_k^2}{2} \Vert v_k \Vert^2] \\
&= \mathbb{E} [f(x_k) - \eta_k ( \nabla f(x_k) - v_k)^\top v_k  + (\frac{L \eta_k^2}{2} - \eta_k) \Vert v_k \Vert^2] \\
&\le  f(x_k) + \frac{ \eta_k}{2} \Vert \nabla f(x_k) - v_k \Vert^2 + (\frac{L \eta_k^2}{2} - \frac{\eta_k}{2}) \Vert v_k \Vert^2  \\
&\le f(x_k) + (\frac{L \eta_k^2}{2} - \frac{\eta_k}{2}) \Vert v_k \Vert^2 + \frac{\eta_k^3}{2 \vert S \vert} \sum_{i=(n_k-1)q}^k \Vert v_i \Vert^2 + \frac{\eta_k \sigma^2}{2}
\end{align*}
$$


继续递推，


$$
\begin{align*}
f(x_{k+1}) \le f(x_{(n_k-1)q}) + \sum_{i=(n_k-1)q}^k (\frac{L \eta_k^2}{2} - \frac{\eta_k}{2} + \frac{ \eta_k^3 q}{2 \vert S \vert}) \Vert v_i \Vert^2 +  \sum_{i = (n_k-1)q}^k \frac{\eta_k \sigma^2}{2} \\
\end{align*}
$$


对于最后一次迭代进行递推，


$$
\begin{align*}
f(x_K) - f(x_0) &\le  \sum_{i=0}^{K-1} (\frac{L \eta_k^2}{2} - \frac{\eta_k}{2} + \frac{ \eta_k^3 q}{2 \vert S \vert}) \Vert v_i \Vert^2 +  \sum_{i = 0}^{K-1} \frac{\eta_k \sigma^2}{2} \\
\end{align*}
$$


移项，


$$
\begin{align*}
\mathbb{E}[ \Vert v_{\xi} \Vert^2] &=\frac{1}{K} \sum_{i=0}^{K-1} \Vert v_i \Vert^2 \le \frac{f(x_0) - f(x_{\ast})}{\beta K} + \frac{\eta_k \sigma^2}{2 \beta} , \beta = \frac{\eta_k}{2} - \frac{L \eta_k^2}{2} - \frac{ \eta_k^3 q}{2 \vert S \vert}
\end{align*}
$$


类似地，


$$
\begin{align*}
\mathbb{E}[ \Vert v_{\xi} - \nabla f(x_{\xi}) \Vert^2] &\le \frac{L^2 \eta_k^2}{\vert S \vert}\sum_{i=(n_\xi - 1)q}^{\xi} \mathbb{E}[\Vert v_i \Vert^2] + \sigma^2 \\
&\le \frac{L^2 \eta_k^2 q}{\vert S \vert }  \frac{1}{K}\sum_{i=0}^{K-1} \Vert v_i \Vert^2 + \sigma^2 \\
&\le \frac{L^2 \eta_k^2 q}{  \beta \vert S \vert K } (f(x_0) - f(x_{\ast})) + \frac{L^2 \eta_k^3 \sigma^2 q}{2 \beta \vert S \vert }
\end{align*}
$$


算法随机选取一个点输出，其梯度 $\mathbb{E}[ \Vert \nabla f(x_{\xi}) \Vert^2]$ 可以有如下估计，


$$
\begin{align*}
\mathbb{E} [ \Vert \nabla f(x_{\xi}) \Vert^2] &\le 2 \mathbb{E} [ \Vert \nabla f(x_{\xi} )  - v_{\xi} \Vert^2] + 2\mathbb{E} [ \Vert v_{\xi} \Vert^2] \\
&\le \frac{2}{ \beta K} (1 + \frac{L^2 \eta_k^2 q}{ \vert S \vert}) (f(x_0) - f(x_{\ast})) + \frac{L^2 \eta_k^3 \sigma^2 q}{\beta \vert S \vert } \\
&= \frac{2}{ \beta K} (1 + L^2 \eta_k^2)(f(x_0 ) - f(x_{\ast})) + \frac{L^2 \eta_k^3 \sigma^2}{\beta} , \beta = \frac{\eta_k}{2} - \frac{L \eta_k^2}{2} - \frac{ \eta_k^3 }{2 }, q = \vert S \vert
\end{align*}
$$


选取如下参数，


$$
\begin{align*}
\eta_k  =\frac{1}{2L}, \beta = \frac{1}{16L}, \sigma^2 = 0
\end{align*}
$$


为了得到 $\epsilon$- 最优解


$$
\begin{align*}
\mathbb{E}[ \Vert \nabla f(x_{\xi}) \Vert^2] &\le \frac{40L}{K} ( f(x_0) - f(x_{\ast})) 
\end{align*}
$$


所需要的计算复杂度为，


$$
\begin{align*}
\mathcal{O}( \lceil \frac{K}{q} \rceil n + K q   ) = \mathcal{O}(n + \sqrt{n} \epsilon^{-2})
\end{align*}
$$

## Online Setting



在线算法中，优化目标函数为更一般的形式，而不一定可以写成有限和的形式，


$$
\begin{align*}
\min_x \mathbb{E}[ f(x)] 
\end{align*}
$$


此时无法计算全梯度，但仍然可以利用采样 $ \vert S' \vert$ 个样本计算梯度，此时相应的方差可以缩小 $\vert S' \vert$ 倍，


$$
\begin{align*}
\mathbb{E} [ \Vert \nabla f(x_{\xi}) \Vert^2] &\le
 \frac{2}{ \beta K} (1 + L^2 \eta_k^2)(f(x_0 ) - f(x_{\ast})) + \frac{L^2 \eta_k^3 \sigma^2}{\beta \vert S' \vert} \\
 &= \frac{40L}{K} (f(x_0) - f(x_{\ast})) + \epsilon^2, \vert S' \vert = \frac{2\sigma^2 }{\epsilon^2}, \eta_k = \frac{1}{2L}
\end{align*}
$$


此时为了达到 $\epsilon$ - 近似解所需要的复杂度为，


$$
\begin{align*}
\mathcal{O}( \epsilon^{-3} + \epsilon^{-2}) = \mathcal{O}(\epsilon^{-3})
\end{align*}
$$
