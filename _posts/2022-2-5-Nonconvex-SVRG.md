---
title: 'Nonconvex SVRG'
toc: true
excerpt_separator: <!--more-->
tags:
 - 优化
 - 随机优化
 - 非凸优化
---

 

论文阅读笔记：[Stochastic variance reduction for nonconvex optimization](https://proceedings.mlr.press/v48/reddi16.html)



<!--more-->



文章关注于SGD和SVRG在非凸优化上的收敛率，假设优化目标函数具有如下有限和的形式，


$$
\begin{align}
\min f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x) 
\end{align}
$$
且每个 $f_i(x)$ 都满足 $L$ -光滑的性质，


$$
\begin{align}
f(x) \le  f(y) + \nabla f(y)^\top(x-y) + \frac{L}{2} \Vert x - y \Vert^2
\end{align}
$$


## Nonconvex SGD



对于随机梯度下降算法（SGD，Stochastic Gradient Descent），算法为，


$$
\begin{align}
g_k &= \nabla f(x_k, \xi) \\
x_{k+1} &= x_k - \eta_k g_k
\end{align}
$$


假设随机梯度的方差上界为 $\sigma^2$, 根据 $L$ -光滑的性质，


$$
\begin{align}
\mathbb{E}[f(x_{k+1})] &\le f(x_k) + \mathbb{E}[\nabla f(x_k)^\top(x_{k+1} - x_k)] + \frac{L}{2 } \mathbb{E}[\Vert x_{k+1} - x_k \Vert^2] \\
&= f(x_k) + \eta_k\mathbb{E} [ \nabla f(x_k)^\top g_k] + \frac{\eta_k^2 L}{2} \mathbb{E} [ \Vert g_k \Vert^2] \\
&\le f(x_k) + ( \frac{\eta_k^2 L}{2}-\eta_k) \Vert \nabla f(x_k ) \Vert^2 + \frac{\eta_k^2 \sigma^2 L}{2} \\
&\le f(x_k) - \frac{\eta_k}{2} \Vert \nabla f(x_k ) \Vert^2 + \frac{\eta_k^2 \sigma^2 L}{2} ,\text{Set } \eta_k \le \frac{1}{L} \\
\end{align}
$$


因此，


$$
\begin{align}
\Vert \nabla f(x_k) \Vert^2 &\le \frac{2}{\eta_k}\mathbb{E} [ f(x_k) - f(x_{k+1})] + \eta_k\sigma^2L \\
\frac{1}{k} \sum_{i=0}^{k-1} \mathbb{E} [\Vert \nabla f(x_k) \Vert^2] &\le \frac{2}{k \eta} \mathbb{E}[f(x_0 ) - f(x_k)] + \eta \sigma^2 L \\
&\le \frac{2}{k \eta} (f(x_0 ) - f(x_{\ast})) + \eta \sigma^2 L \\
&= 2\sqrt{\frac{2\sigma^2 L(f(x_0 ) - f(x_{\ast}))}{k }} \\
\text{Set } \eta  &= \sqrt{\frac{2 (f(x_0) - f(x_{\ast}))}{k \sigma^2 L}}
\end{align}
$$




为了达到平稳点的 $\epsilon$ -近似解，需要的计算复杂度为，


$$
\begin{align}
T = \mathcal{O}(\frac{1}{\epsilon^2})
\end{align}
$$


## Nonconvex SVRG



考虑方差缩减技术 [SVRG](https://truenobility303.github.io/L-SVRG-and-L-Katyusha/)， 其算法为每隔 $m$ 轮计算一次全梯度 $\nabla f(\tilde x)$, 


$$
\begin{align}
g_k &= \nabla f(x_k, \xi) - \nabla f(\tilde x, \xi) + \nabla f(\tilde x) \\
x_{k+1} &= x_k - \eta_k g_k
\end{align}
$$


此时随机梯度的方差可以被控制住，


$$
\begin{align}
Var[g_k] &= \mathbb{E}[ \Vert g_k - \nabla f(x_k) \Vert^2] \\
&= \mathbb{E}[ \Vert \nabla f(x_k, \xi) - \nabla f(\tilde x, \xi) + \nabla f(\tilde x) - \nabla f(x_k) \Vert^2] \\
&\le \mathbb{E} [\Vert \nabla f(x_k,\xi) - \nabla f(\tilde x, \xi) \Vert^2] \\
&\le L^2 \Vert x_k - \tilde x \Vert^2
\end{align}
$$




同理，根据 $L$ -光滑的性质，


$$
\begin{align}
\mathbb{E}[f(x_{k+1})] &\le f(x_k) + \mathbb{E}[\nabla f(x_k)^\top(x_{k+1} - x_k)] + \frac{L}{2 } \mathbb{E}[\Vert x_{k+1} - x_k \Vert^2] \\
&\le f(x_k) +(\frac{\eta_k^2L}{2} - \eta_k) \Vert \nabla f(x_k ) \Vert^2 + \frac{\eta_k^2 L}{2} Var[g_k] ,\text{Set } \eta_k \le \frac{1}{L} \\
\end{align}
$$



对于距离，



$$
\begin{align}
\mathbb{E} [\Vert x_{k+1} - \tilde x \Vert^2] &= \mathbb{E}[ \Vert x_k - \eta_k g_k  - \tilde x \Vert^2] \\
&\le \Vert x_k - \tilde x \Vert^2 - \eta_k \mathbb{E}[g_k^\top(x_k - \tilde x)] + \eta_k^2 \mathbb{E}[ \Vert g_k \Vert^2] \\
&=\Vert x_k - \tilde x \Vert^2 - 2\eta_k \nabla f(x_k)^\top (x_k - \tilde x) + \eta_k^2 \Vert \nabla f(x_k) \Vert^2 + \eta_k^2 Var[g_k] \\
&\le (1 +\beta_k \eta_k)\Vert x_k - \tilde x \Vert^2 + (\eta_k^2 + \frac{\eta_k}{\beta_k})  \Vert \nabla f(x_k) \Vert^2 + \eta_k^2 Var[g_k]
\end{align}
$$



定义Lyapunov函数，


$$
\begin{align}
\mathbb{E}[\mathcal{V}_{k+1}] &= \mathbb{E}[f(x_{k+1}) - f(x_{\ast}) + c_{k+1} \Vert x_{k+1} - \tilde x \Vert^2 ]\\
&\le f(x_k) - f(x_{\ast}) + (\frac{\eta_k^2L}{2} - \eta_k) \Vert \nabla f(x_k ) \Vert^2 + \frac{\eta_k^2 L}{2} Var[g_k] \\
&\quad + (1 +\beta_k \eta_k) c_{k+1} \Vert x_k - \tilde x \Vert^2 + (\eta_k^2 + \frac{\eta_k}{\beta_k}) c_{k+1} \Vert \nabla f(x_k) \Vert^2 + \eta_k^2 c_{k+1}Var[g_k] \\
&\le f(x_k) - f(x_{\ast} ) + (\frac{\eta_k^2L}{2} - \eta_k + c_{k+1}\eta_k^2 +\frac{c_{k+1}\eta_k}{\beta_k} ) \Vert \nabla f(x_k) \Vert^2 +(c_{k+1}+ c_{k+1}\beta_k \eta_k + c_{k+1}\eta_k^2 L^2 + \frac{\eta_k^2L^3}{2}) \Vert x_k - \tilde x \Vert^2
\end{align}
$$


如果选取合适的参数使得，


$$
\begin{align}
c_{k} &= c_{k+1}+ c_{k+1}\beta_k \eta_k + c_{k+1}\eta_k^2 L^2 + \frac{\eta_k^2L^3}{2} \\
\gamma_k &= \eta_k - \frac{\eta_k^2L}{2} - c_{k+1}\eta_k^2 -\frac{c_{k+1}\eta_k}{\beta_k} \ge 0 \\
\end{align}
$$



对于第 $s$ 轮方差缩减中的第 $k$ 次迭代，



$$
\begin{align}
\mathbb{E}[\mathcal{V}_{s,k+1}] &\le \mathcal{V_{s,k}} - \gamma_{s,k} \Vert \nabla f(x_{s,k}) \Vert^2 \\
\Vert \nabla f(x_{s,k}) \Vert^2 &\le \frac{\mathbb{E}[\mathcal{V_{s,k}} - \mathcal{V}_{s,k+1}]}{\gamma_{s,k}} \\
\frac{1}{m} \sum_{k=0}^{m-1}\mathbb{E}[\Vert \nabla f(x_{s,k}) \Vert^2] &\le \frac{\mathbb{E}[\mathcal{V}_{s,0} - \mathcal{V}_{s,m} ]}{m\gamma} , \text{Let } \gamma = \min_k \gamma_{s,k}  \\
\frac{1}{m} \sum_{k=0}^{m-1}\mathbb{E}[\Vert \nabla f(x_{s,k}) \Vert^2] &\le \frac{\mathbb{E}[\mathcal{V}_{s} - \mathcal{V}_{s+1} ]}{m\gamma} ,\text{By } \mathcal{V}_s = \mathcal{V}_{s,0} = \mathcal{V}_{s-1,m}\\
\frac{1}{Sm} \sum_{s=0}^{S-1} \sum_{k=0}^{m-1}\mathbb{E}[\Vert \nabla f(x_{s,k}) \Vert^2] &\le \frac{\mathbb{E}[\mathcal{V}_{0} - \mathcal{V}_{S} ]}{Sm\gamma} \le \frac{\mathcal{V_0}}{Sm \gamma} = \frac{f(x_0) - f(x_{\ast})}{Sm \gamma}\\
\end{align}
$$



下面选取参数，首先观察到，

$$
\begin{align}
c_{k} &= \theta c_{k+1} + \frac{\eta_k^2L^3}{2} > c_{k+1},\text{Let } \theta = \beta_k \eta_k + \eta_k^2 L^2 \\ 
c_0 &= \frac{\eta_k^2L^3}{2} \frac{(1 + \theta)^m - 1}{ \theta} \\
\gamma &\ge \eta_k - \frac{\eta_k^2 L}{2} - c_0  \eta_k^2 -\frac{c_0 \eta_k}{ \beta_k}
\end{align}
$$



令步长与 $n$ 相关，


$$
\begin{align}
\eta_k &= \frac{k_1}{L n^\alpha}, \beta_k = \frac{k_2 L}{n^{\beta}} ,\text{with } k_1,k_2 \le 1 ,\beta  \le\alpha\\
\theta &=  \beta_k \eta_k + \eta_k^2 L^2  = \frac{k_1k_2}{n^{\alpha+ \beta}} + \frac{k_1^2}{n^{2 \alpha}} \ge \frac{2k_1}{n^{\alpha + \beta}}\\ 
c_0 &\le  \frac{k_1^2 L(e-1)}{ 2 n^{2 \alpha}\theta} \le  \frac{k_1 L(e-1)}{4 n^{\alpha - \beta}} ,\text{ with } m = \frac{n^{\alpha + \beta}}{2k_1} \\
c_0 \eta_k &\le \frac{k_1^2 (e-1)}{4n^{2\alpha - \beta}},  \frac{c_0}{\beta_k} \le \frac{k_1(e-1)}{k_2 n^{\alpha - 2 \beta}} ,\text{ with } \beta \le \frac{\alpha}{2}  \\
\gamma &\ge  ( \frac{1}{2} - \frac{k_1(e-1)}{4} - \frac{k_1 (e-1)}{k_2}) \eta_k \ge \frac{1}{4} \eta_k =\frac{k_1}{4 L n^{\alpha}} , \text{with } k_1 \le \frac{1}{5(e-1)}, k_2 =1 \\
\end{align}
$$


为了达到 $\epsilon$- 最优解的计算复杂度，


$$
\begin{align}
Sm &= \frac{\gamma}{\epsilon} (f(x_0) - f(x_{\ast})) \le \frac{4 L n^{\alpha}}{k_1 \epsilon} (f(x_0) - f(x_{\ast}))  \\
S &= \frac{\gamma}{m\epsilon} (f(x_0) - f(x_{\ast})) \le \frac{2L}{k_1^2 n^{\beta}\epsilon} (f(x_0) - f(x_{\ast})) \\
T &= n +S(m +n) \le  n+ \frac{4 L (n^{\alpha} + n^{1 -\beta})}{k_1 \epsilon} \\
\end{align}
$$

当 $\alpha = 1 -\beta$ 取得最优的复杂度，并且考虑到约束 $\beta \le \frac{\alpha}{2}$ ，得到，



$$
\begin{align}
T = \mathcal{O}( n+ \frac{n^{\frac{2}{3}}L}{\epsilon}) , \text {with } \alpha = \frac{2}{3} , \beta = \frac{1}{3}
\end{align}
$$



## Gradient Dominant Case



当优化目标函数满足 PL条件，也称为 Gradient Dominant Fucntion，也即满足非凸性不那么强的时候，


$$
\begin{align}
 \Vert \nabla f(x) \Vert^2 \ge 2 \mu (f(x_k) - f(x_{\ast}))
\end{align}
$$


使用带重启动（Restart）策略的 SVRG算法，也即每隔 $N$ 轮在之前的轮次中随机选择一个位置作为下一轮次的初始点，


$$
\begin{align}
\frac{1}{Sm} \sum_{s=0}^{S-1} \sum_{k=0}^{m-1}\mathbb{E}[\Vert \nabla f(x_{n,s,k}) \Vert^2] &\le \frac{\mathbb{E}[\mathcal{V}_{n,0} - \mathcal{V}_{n,S} ]}{Sm\gamma} \le \frac{4L n^{\frac{2}{3}}\mathbb{E}[\mathcal{V}_{n,0} - \mathcal{V}_{n,S} ]}{k_1 Sm } \le \frac{4L n^{\frac{2}{3}}\mathbb{E}[\mathcal{V}_{n} ]}{k_1 Sm }\\
\end{align}
$$


根据重启动策略，


$$
\begin{align}
\mathbb{E}[\mathcal{V}_{n+1} ] = \mathbb{E}[f(x_{n+1}) - f(x_{\ast})] \le \frac{1}{2 \mu} \mathbb{E} [\Vert \nabla f(x_{n+1}) \Vert^2] \le \frac{2L n^{\frac{2}{3}} \mathcal{V_n}}{k_1 \mu Sm} \le \frac{1}{2} \mathcal{V_n}, \text{Let } Sm = \frac{4L n^{\frac{2}{3}}}{k_1 \mu},S = \frac{8 L }{\mu n^{\frac{1}{3}}}
\end{align}
$$


因此达到 $\epsilon$ -最优解需要的计算复杂度为，


$$
\mathcal{O}(T(Sm+n) = \mathcal{O}((Sm +n)\log \frac{1}{\epsilon}) = \mathcal{O}((n+ n^{\frac{2}{3}}\kappa) \log \frac{1}{\epsilon}), \text{with } \kappa  = \frac{L}{\mu}
$$
 