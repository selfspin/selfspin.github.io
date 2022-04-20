---
title: 'Katyusha X'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 随机优化
  - 非凸优化
---



论文阅读笔记：[Katyusha X: Practical Momentum Method for Stochastic Sum-of-Nonconvex Optimization](https://arxiv.org/abs/1802.03866)



<!--more-->



KatyushaX 将Katyusha推广到每一个样本非凸但是目标函数仍然为凸函数的情况



## Mirror Descent



考虑使用Mirror Desent算法应用于 $\sigma_{\psi}$- 强凸函数，对于如下 $z_{k+1}$,



$$
\begin{align*}
z_{k+1} &= \text{argmin}_z \frac{1}{2\alpha} \Vert z - z_k \Vert^2 + g_k^\top z + h(z),
\end{align*}
$$
利用最优性条件，
$$
\begin{align*}
0 &= \frac{1}{\alpha} (z_{k+1} - z_k) + g_k + \nabla h(z_{k+1}).
\end{align*}
$$


因此，
$$
\begin{align*}
&\quad g_k^\top (z_k - u) + h(z_{k+1}) - h(u)\\ &= g_k^\top (z_k - z_{k+1}) + g_k^\top (z_{k+1} - u) + h(z_{k+1}) - h(u)\\
&\le g_k^\top (z_k - z_{k+1}) + g_k^\top (z_{k+1} - u) - \nabla h(z_{k+1})^\top (u - z_{k+1}) - \frac{\sigma_{\psi}}{2} \Vert  u - z_{k+1} \Vert^2 \\
&=g_k^\top (z_k - z_{k+1}) + (g_k +\nabla h(z_{k+1})^\top (z_{k+1} - u) - \frac{\sigma_{\psi}}{2} \Vert  u - z_{k+1} \Vert^2 \\
&= g_k^\top (z_k - z_{k+1}) +\frac{1}{\alpha} (z_k - z_{k+1})^\top (z_{k+1} - \sigma_{f})  - \frac{\sigma_{\psi}}{2} \Vert  u - z_{k+1} \Vert^2 \\
&= g_k^\top (z_k - z_{k+1}) + \frac{1}{2\alpha} \Vert z_k - u \Vert^2 - \frac{1}{2\alpha} \Vert z_k - z_{k+1} \Vert^2 - \frac{1+ \sigma_{\psi} \alpha}{2\alpha} \Vert z_{k+1} - u \Vert^2 \\
&\le \frac{\alpha}{2} \Vert g_k \Vert^2  + \frac{1}{2\alpha} \Vert z_k - u \Vert^2 - \frac{1+ \sigma_{\psi} \alpha}{2\alpha} \Vert z_{k+1} - u \Vert^2, \forall u.
\end{align*}
$$



## SVRG 



对于梯度的无偏估计量 $\tilde \nabla  f(w_k)$, 使用梯度下降，


$$
\begin{align*}
w_{k+1} = \text{argmin}_y \frac{1}{2 \eta} \Vert y - w_k \Vert^2 + \tilde \nabla f(w_k)^\top y + \psi (y).
\end{align*}
$$

对于 $\sigma_{f}$ -强凸并且 $L$- 光滑的函数，成立



$$
\begin{align*}
&\quad\mathbb{E} [ f(w_{k+1} ) + \psi(w_{k+1}) - f(u) - \psi(u)] \\
& \le \mathbb{E}[f(w_k) + \nabla f(w_k)^\top (w_{k+1} - w_k) + \frac{L}{2} \Vert w_k - w_{k+1} \Vert^2  - f(u) + \psi (w_{k+1} ) - \psi (u)] \\
&\le  \mathbb{E}[\nabla f(w_k)^\top (w_{k+1} - w_k) + \frac{L}{2} \Vert w_k - w_{k+1} \Vert^2 - \nabla f(w_k)^\top ( u - w_k) - \frac{\sigma_{f}}{2} \Vert  u - w_k \Vert^2 + \psi (w_{k+1} ) - \psi (u)] \\
&\le \mathbb{E}[\tilde \nabla f(w_k)^\top ( w_k - u) + \nabla f(w_k)^\top (w_{k+1} - w_k) + \frac{L}{2} \Vert w_k - w_{k+1} \Vert^2  - \frac{\sigma_{f}}{2} \Vert  u - w_k \Vert^2 + \psi (w_{k+1} ) - \psi (u) ] \\
& \le \mathbb{E} [(\nabla f(w_k) - \tilde \nabla f(w_k)^\top (w_{k+1}-w_k)+ \frac{1- \sigma_{f} \eta}{2\eta} \Vert w_{k} - u \Vert^2  - \frac{1 -L \eta}{2\eta} \Vert w_k - w_{k+1} \Vert^2 - \frac{1+\sigma_{\psi} \eta}{2\eta} \Vert w_{k+1}- u \Vert^2  ] \\
&\le \mathbb{E} [ \frac{\eta}{2(1-L\eta)} \Vert \tilde \nabla  f(w_k) - \nabla f(w_k) \Vert^2 + \frac{1 - \sigma_{f} \eta}{2\eta} \Vert w_k - u \Vert^2 -  \frac{1+\sigma_{\psi} \eta}{2\eta} \Vert w_{k+1}- u \Vert^2  ].
\end{align*}
$$



每一个Epoch的数目从几何分布中采样，可以验证当 $M$ 服从参数为 $1/m$ 的几何分布时对于序列 $D_M$ 具有如下性质，



$$
\begin{align*}
\mathbb{E}_M[D_M]  = \mathbb{E} [ (1-1/m) D_{M+1} + 1/m D_0]
\end{align*}
$$



使用方差缩减技术过后，令 $ m \ge 2, \eta \le \frac{1}{2L}  \min \{ 1, \sqrt{\frac{b}{m}} \} $, 则



$$
\begin{align*}
&\quad\mathbb{E} [ f(w_{M+1} ) + \psi(w_{M+1}) - f(u) - \psi(u)] \\
&\le \mathbb{E} [ \eta\Vert \tilde \nabla  f(w_{M}) - \nabla f(w_{M}) \Vert^2 + \frac{1 - \sigma_{f} \eta}{2\eta} \Vert w_{M} - u \Vert^2 -  \frac{1+\sigma_{\psi} \eta}{2\eta} \Vert w_{M+1}- u \Vert^2  ]\\ 
&\le \mathbb{E} [\frac{L^2 \eta}{b} \Vert w_{M} - w_0 \Vert^2 +  \frac{1 - \sigma_{f} \eta}{2\eta} \Vert w_{M} - u \Vert^2 -  \frac{1+\sigma_{\psi} \eta}{2\eta} \Vert w_{M+1}- u \Vert^2  ]\\
&= \mathbb{E}[ \frac{L^2(m-1)\eta}{bm} \Vert w_{M+1} - w_0 \Vert^2 + \frac{ \Vert w_0 -  u  \Vert^2  - \Vert w_M - u \Vert^2}{2 \eta(m-1)} - \frac{\sigma_{f}}{2} \Vert w_M - u \Vert^2 - \frac{\sigma_{\psi}}{2} \Vert w_{M+1} - u \Vert^2 ] \\
&= \mathbb{E}[\frac{L^2(m-1)\eta}{bm} \Vert w_{M+1} - w_0 \Vert^2 + \frac{ \Vert w_0 -  u  \Vert^2  - \Vert w_{M+1} - u \Vert^2}{2 m\eta} -  \frac{\sigma_{f}}{2} \Vert w_M - u \Vert^2 - \frac{\sigma_{\psi}}{2} \Vert w_{M+1} - u \Vert^2  ] \\ 
&=\mathbb{E}[\frac{L^2(m-1)\eta}{bm} \Vert w_{M+1} - w_0 \Vert^2 + \frac{ \Vert w_0 -  u  \Vert^2  - \Vert w_{M+1} - u \Vert^2}{2 m\eta} -  \frac{(m-1)\sigma_{f}}{2m} \Vert w_{M+1} - u \Vert^2  - \frac{\sigma_{f}}{2m} \Vert w_0 - u \Vert^2 -  \frac{\sigma_{\psi}}{2} \Vert w_{M+1} - u \Vert^2 ] \\ 
&\le \mathbb{E} [ \frac{L^2\eta}{b} \Vert w_{M+1 } - w_0 \Vert^2 +  \frac{ \Vert w_0 -  u  \Vert^2  - \Vert w_{M+1} - u \Vert^2}{2 m\eta}  - \frac{\sigma_{f} + \sigma_{\psi}}{4} \Vert w_{M+1} - u \Vert^2] \\
&\le \mathbb{E}[ \frac{\Vert w_{M+1} - w_0 \Vert^2 + \Vert w_ 0 - u \Vert^2 - \Vert w_{M+1} - u \Vert^2 }{2m\eta} - \frac{\sigma_{f} + \sigma_{\psi}}{4} \Vert w_{M+1} - u \Vert^2] \\
&= \mathbb{E}[ -\frac{ \Vert w_{M+1} - w_0 \Vert^2}{4m \eta} + \frac{ \Vert w_0 -u \Vert^2 - \Vert w_{M+1} - w_0 \Vert^2 - \Vert w_{M+1} - u \Vert^2}{2m\eta} - \frac{\sigma_{f} + \sigma_{\psi}}{4} \Vert w_{M+1} - u \Vert^2] \\
&= \mathbb{E}[ -\frac{ \Vert w_{M+1} - w_0 \Vert^2}{4m \eta}  + \frac{ (w_{M+1} - w_0)^\top(w_ 0 -u)}{m \eta} -\frac{\sigma_{f} + \sigma_{\psi}}{4} \Vert w_{M+1} - u \Vert^2]
\end{align*}
$$



## Framework of Katyusha-X



利用 [Linear Couping](https://truenobility303.github.io/Linear-Coupling/) 的技巧， 得到如下的Katyusha-X的算法，
$$
\begin{align*}
x_{k+1} &= \tau_k z_k  + (1-\tau_k) y_k \\
y_{k+1} &= \text{SVRG} (f + \psi,x_{k+1}) \\
g_{k+1} &= \frac{x_{k+1} - y_{k+1}}{m \eta} \\
z_{k+1} &= \text{argmin}_z  \{ \frac{1}{2\alpha_{k+1}} \Vert z - z_k \Vert^2 + g_{k+1}^\top z + \frac{\mu}{4} \Vert z  - y_{k+1} \Vert^2\}.
\end{align*}
$$

因此，
$$
\begin{align*}
&\quad\mathbb{E}[\alpha_{k+1} ( f(y_{k+1}) + \psi (y_{k+1})- f(u) - \psi (u))] \\
&\le \mathbb{E} \left[-\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 +\alpha_{k+1}g_{k+1}^\top (x_{k+1} - u) - \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - u \Vert^2\right ] \\
&\le \mathbb{E} \left[-\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 +\alpha_{k+1}g_{k+1}^\top (x_{k+1} - z_k) + \alpha_{k+1} g_{k+1}^\top (z_k - u)- \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - u \Vert^2\right ] \\
&= \mathbb{E} \left[-\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 +\frac{(1-\tau_k)\alpha_{k+1}}{\tau_k}g_{k+1}^\top (y_k - x_{k+1} ) + \alpha_{k+1} g_{k+1}^\top (z_k - u)- \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - u \Vert^2\right ] \\
&\le \mathbb{E} \left[\frac{\alpha_{k+1}^2}{2} \Vert g_{k+1} \Vert^2  + \frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+  \alpha_{k+1} \mu/2}{2} \Vert z_{k+1} - u \Vert^2  \right] \\
&\quad  + \mathbb{E} \left[-\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 +\frac{(1-\tau_k)\alpha_{k+1}}{\tau_k}g_{k+1}^\top (y_k - x_{k+1} ) - \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - z_{k+1} \Vert^2\right ] \\
&\le \mathbb{E} \left[\frac{\alpha_{k+1}^2}{2} \Vert g_{k+1} \Vert^2  + \frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+  \alpha_{k+1} \mu/2}{2} \Vert z_{k+1} - u \Vert^2  \right] \\
&\quad  + \mathbb{E} \left[-\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 - \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - z_{k+1} \Vert^2\right ] \\
&\quad + \mathbb{E} \left[ \frac{(1-\tau_k)\alpha_{k+1}}{\tau_k} \left( f(y_k) + \psi(y_k)  - f(y_{k+1}) - \psi(y_{k+1})   -\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 - \frac{\alpha_{k+1}\mu}{4} \Vert y_{k+1} - y_k \Vert^2\right) \right ] \\
&\le \mathbb{E} \left[\frac{\alpha_{k+1}^2}{2} \Vert g_{k+1} \Vert^2  + \frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+  \alpha_{k+1} \mu/2}{2} \Vert z_{k+1} - u \Vert^2  -\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2\right] \\
&\quad + \mathbb{E} \left[ \frac{(1-\tau_k)\alpha_{k+1}}{\tau_k} \left( f(y_k) + \psi(y_k)  - f(y_{k+1}) - \psi(y_{k+1})   -\frac{\alpha_{k+1}m \eta}{4} \Vert g_{k+1} \Vert^2 \right) \right ] \\
\end{align*}
$$


令 $F = f + \psi$,  移项后得到，
$$
\begin{align*}
\mathbb{E} \left[ \frac{\alpha_{k+1}}{\tau_{k}} \left( F(y_{k+1}) - F(u)\right)\right ] &\le \mathbb{E}\left[ \frac{(1-\tau_k)\alpha_{k+1}}{\tau_k} \left( F(y_k) - F(u) \right)\right] + \mathbb{E} \left[ \left(\frac{\alpha_{k+1}^2}{2} - \frac{\alpha_{k+1} m \eta}{4 \tau_k}\right) \Vert g_{k+1} \Vert^2 \right ] \\
&\quad + \mathbb{E}\left[\frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+\alpha_{k+1} \mu /2}{2} \Vert z_{k+1} - u \Vert^2 \right]
\end{align*}
$$



## Strongly-convex Case



选择以下参数，
$$
\begin{align*}
\eta &= \frac{1}{2 L\sqrt{n} }, m = n ,\\ 
\tau_k &= \tau = \frac{\sqrt{m \eta \mu}}{2} =  \frac{1}{2} \sqrt{\frac{\mu \sqrt{n} }{2L}}, \\
\alpha_{k+1} &= \frac{m \eta}{2 \tau} =\frac{2 \tau }{\mu} = \sqrt{\frac{\sqrt{n} }{2 \mu L}}.
\end{align*}
$$
代入上述参数后可以得到，
$$
\begin{align*}
\mathbb{E} \left[ \frac{2}{\mu} ( F(y_{k+1} ) - F(x^{\ast}))\right] &\le \mathbb{E} \left[ \frac{2(1- \tau)}{\mu} (F(y_k) - F(x^{\ast}))\right] + \mathbb{E}\left[\frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+\tau}{2} \Vert z_{k+1} - x^{\ast} \Vert^2 \right] \\
&\le \mathbb{E} \left[ \frac{2}{(1+\tau)\mu} (F(y_k) - F(x^{\ast}))\right] + \mathbb{E}\left[\frac{1}{2} \Vert z_k - u \Vert^2 - \frac{1+\tau}{2} \Vert z_{k+1} - x^{\ast} \Vert^2 \right] \\
\end{align*}
$$
移项后得到，
$$
\begin{align*}
\mathbb{E} \left[ \frac{2}{\mu} ( F(y_{k+1} ) - F(x^{\ast})) + \frac{1+ \tau}{2} \Vert z_{k+1} - x^{\ast} \Vert^2\right] &\le \frac{1}{1+\tau} \mathbb{E} \left[  \frac{2}{\mu} ( F(y_{k} ) - F(x^{\ast})) + \frac{1+ \tau}{2} \Vert z_{k} - x^{\ast} \Vert^2 \right].
\end{align*}
$$


算法的复杂度为，
$$
\begin{align*}
K & \le \left( \frac{1}{\tau}  +1\right) n \log \frac{1}{\epsilon} \\
&\le \left(\frac{n}{\tau} +n\right) \log \frac{1}{\epsilon} \\
&= \left( 2 \sqrt{2} \times  n^{3/4} \sqrt{\kappa} +n\right) \log \frac{1}{\epsilon} \\
&= \mathcal{O}( n + n^{3/4} \sqrt{\kappa}) \log \frac{1}{\epsilon}.
\end{align*}
$$


如果使用Minibatch，可以选择，
$$
\begin{align*}
\eta &= \frac{1}{2 L}, m = \sqrt{n} , b = \sqrt{n}\\ 
\tau_k &= \tau = \frac{\sqrt{m \eta \mu}}{2} =  \frac{1}{2} \sqrt{\frac{\mu \sqrt{n} }{2L}}, \\
\alpha_{k+1} &= \frac{m \eta}{2 \tau} =\frac{2 \tau }{\mu} = \sqrt{\frac{\sqrt{n} }{2 \mu L}}.
\end{align*}
$$
递推关系式仍然成立，此时算法的复杂度同样为，
$$
\begin{align*}
T & \le \left( \frac{1}{\tau}  +1\right) n \log \frac{1}{\epsilon} = \mathcal{O}( n + n^{3/4} \sqrt{\kappa}) \log \frac{1}{\epsilon}.
\end{align*}
$$


## Convex Case



当目标函数不满足强凸的假设时，选择的参数为，


$$
\begin{align*}
\eta &= \frac{1}{2 L\sqrt{n} }, m = n ,\tau_k = \frac{2}{k+2}，\\ 
\alpha_{k+1} &= \frac{m \eta}{2 \tau_k} =  \frac{\sqrt{n}}{8L(k+2)}.
\end{align*}
$$


将上述参数代入前述式子可以得到，
$$
\begin{align*}
\mathbb{E} \left[ \frac{m \eta}{2\tau_{k}^2} \left( F(y_{k+1}) - F(x^{\ast})\right)\right ] &\le \mathbb{E}\left[ \frac{(1-\tau_k) m \eta}{2\tau_k^2} \left( F(y_k) - F(x^{\ast}) \right)\right] 
 + \mathbb{E}\left[\frac{1}{2} \Vert z_k - x^{\ast} \Vert^2 - \frac{1}{2} \Vert z_{k+1} - x^{\ast} \Vert^2 \right] \\
 &\le \mathbb{E}\left[ \frac{ m \eta}{2\tau_{k-1}^2} \left( F(y_k) - F(x^{\ast}) \right)\right] 
 + \mathbb{E}\left[\frac{1}{2} \Vert z_k - x^{\ast} \Vert^2 - \frac{1}{2} \Vert z_{k+1} - x^{\ast} \Vert^2 \right] \\
\end{align*}
$$

移项后得到，
$$
\begin{align*}
\mathbb{E} \left[ \frac{m \eta}{2\tau_{k}^2} \left( F(y_{k+1}) - F(x^{\ast})\right) + \frac{1}{2} \Vert z_{k+1} - z^{\ast} \Vert^2 \right ] &\le \mathbb{E} \left[ \frac{m \eta}{2\tau_{k-1}^2} \left( F(y_{k}) - F(x^{\ast})\right) + \frac{1}{2} \Vert z_{k} - z^{\ast} \Vert^2 \right ] 
\end{align*}
$$
类似地可以得到计算复杂度为，
$$
\begin{align*}
T &\le n + \frac{16  n^{3/4} \sqrt{L}}{ \sqrt{\epsilon}} = \mathcal{O} \left( n + \frac{  n^{3/4} \sqrt{L}}{ \sqrt{\epsilon}} \right). 
\end{align*}
$$
