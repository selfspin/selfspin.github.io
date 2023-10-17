---
title: 'First-Order Method for Bilevel Optimization under PL condition'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - 优化
  - 双层优化

---



Paper Reading: On Penalty Methods for Nonconvex Bilevel Optimization and First-Order Stochastic Approximation.



<!--more-->



## Introduction

文章考虑双层优化问题，



$$
\min_{x,y \in Y^\ast(x)} f(x,y) \quad{\rm s.t.} \quad Y^\ast(x) = \arg \min_{y \in \mathcal{Y}} g(x,y).
$$



一种主流的方法考虑将收敛指标定义为关于如下的hyper-objective的收敛 [4]，



$$
\begin{align*}
\min_x \varphi(x), \quad {\rm where} \quad \varphi(x) = \min_{y \in Y^\ast(x)} f(x,y)
\end{align*}
$$

近期，不少研究聚焦于Penalty方法 [3]，也即给定足够小的扰动项 $\sigma$, 求解如下的无约束问题



$$
\begin{align*}
\min_{x,y \in \mathcal{Y}} \sigma f(x,y) + g(x,y) - g^\ast(x), \quad {\rm where} \quad g^\ast(x) = \min_{y \in \mathcal{Y}} g(x,y).
\end{align*}
$$

定义



$$
\begin{align*}
l(x,\sigma) = \min_{y \in \mathcal{Y}} \{h_{\sigma}(x,y):=\sigma f(x,y) + g(x,y) \}
\end{align*}
$$

我们希望如下定义的penalized hyper-objective



$$
\begin{align*}
\varphi_{\sigma} := \frac{l(x,\sigma) - l(x,0)}{\sigma}  = \min_y \left\{ f(x,y) + \frac{g(x,y) - g^\ast(x)}{\sigma} \right\}
\end{align*}
$$



可以是原问题的一个有效近似。两者的关系在无约束强凸的情形下已经被研究的比较透彻，据此 [2] 提出了第一个对于双层问题具有严格收敛保证的一阶算法，而 [5] 在其基础上提出了具有近似最优复杂度的一阶算法。本文聚焦于如下满足Proximal EB 条件的函数类，



$$
\begin{align*}
\rho^{-1} \Vert y - {\rm prox}_{\rho h_{\sigma}(x,\,\cdot\,) } (y) \Vert \ge \mu ~{\rm dist}(y,Y_{\sigma}^\ast(x)),\quad {\rm where} \quad Y_{\sigma}^\ast(x) = \arg \min_{y \in \mathcal{Y}} h_{\sigma}(x,y) 
\end{align*}
$$



可以验证强凸函数满足上述性质，因此本文考虑的函数类比经典的强凸假设更广。

在上述条件下，可以证明 $Y_{\sigma}^\ast(x)$ 关于 $x$ 和 $\sigma$ 都具有Lipschitz性质，以下证明与 [5] 的结论类似。



对于任意的 $y_1 \in Y_{\sigma_1}^\ast(x_1)$, 都存在距离最近的 $y_2 \in Y_{\sigma_2}^\ast(x_2)$, 使得


$$
\begin{align*}
&\quad \mu \Vert y_1 - y_2 \Vert \\
&\le \rho^{-1} \Vert  y_1 -{\rm prox}_{\rho h_{\sigma_1} (x_2,\,\cdot\,)}(y_1) \Vert \\
&=  \rho^{-1} \Vert  {\rm prox}_{\rho h_{\sigma_2} (x_2,\,\cdot\,)}(y_1) -{\rm prox}_{\rho h_{\sigma_1} (x_1,\,\cdot\,)}(y_1) \Vert \\
&\le \mathcal{O} (L_g) \Vert x_1 - x_2 \Vert + \mathcal{O} (L_f) \Vert \sigma_1 - \sigma_2 \Vert.
\end{align*}
$$


根据 $y_1,y_2$ 的对称性，可以得到


$$
\begin{align*}
{\rm dist}(Y_{\sigma_1}^\ast(x_1), Y_{\sigma_2}^\ast(x_2)) \le  \mathcal{O}(L_g) \Vert x_1 - x_2 \Vert + \mathcal{O}(L_f) \Vert \sigma_1 - \sigma_2 \Vert.
\end{align*}
$$



上述性质的直接推论是 ${\rm dist}( Y_{\sigma}^\ast(x), Y^\ast(x)) \le \mathcal{O}(\sigma L_f / \mu) $.



首先根据定义我们可以简单验证 $\varphi_{\sigma}(x) \le \varphi(x)$. 进一步，推广 [5] 中关于强凸无约束情况下的分析



对于任意的 $y_{\sigma}^\ast(x) \in Y_{\sigma}^\ast(x)$, 都存在距离最近的 $y^\ast(x) \in Y^\ast(x)$, 使得



$$
\begin{align*}
&\quad \varphi(x) - \varphi_{\sigma}(x) \\
&\le f(x,y^\ast(x)) - f(x,y_{\sigma}^\ast(x)) + \sigma^{-1}  (g(x,y_{\sigma}^\ast(x)) - g(x,y^\ast(x)) ) \\
&\le  C_f \Vert y_{\sigma}^\ast(x) - y^\ast(x) \Vert  + \frac{L_g}{2\sigma} \Vert y_{\sigma}^\ast(x) - y^\ast(x) \Vert^2 \\
&\le \mathcal{O}\left( \frac{C_f L_f}{\mu} + \frac{L_g L_f^2}{\mu^2} \right) \times \sigma.
\end{align*}
$$


这告诉我们


$$
\begin{align*}
\lim_{\sigma \rightarrow 0^+} \varphi_{\sigma}(x) = \frac{\partial }{\partial \sigma} l(x,\sigma) \mid_{\sigma = 0^+} = \varphi(x).
\end{align*}
$$





但对于优化问题，我们通常希望寻求 $\varphi(x)$ 的一个驻点，此时函数的逐点收敛并不足够，我们希望下面的式子满足，



$$
\begin{align*}
\lim_{\sigma \rightarrow 0^+} \nabla \varphi_{\sigma}(x) =  \frac{\partial^2 }{\partial \sigma\partial x } l(x,\sigma) \mid_{\sigma = 0^+} \overset{?}{=}   \frac{\partial^2 }{\partial x  \partial \sigma} l(x,\sigma) \mid_{\sigma = 0^+} = \nabla \varphi(x).
\end{align*}
$$



我们发现问题的关键在于证明函数 $l(x,\sigma)$ 二阶可导，

注意到 $l(x,\sigma)$ 是一个价值函数 (value function), 令 $w=(x,\sigma)$ 以及 $y=\theta$， 上述问题转化为对于 $v(w) = \min_{\theta} p(w,\theta) $ 的二阶可导性问题。

遗憾的是，考虑 [5] 中给出的例子，上述关系式in general并不满足，因此我们需要考虑引入额外的条件。首先，假设约束集合 $\mathcal{Y}$ 是一个满足以下的代数性质的凸集



$$
\begin{align*}
\mathcal{Y} = \{y: h_i(y) \le 0 \}, \quad i = 1,\cdots,m.
\end{align*}
$$



利用上述的代数性质可以定义如下的Lagrange函数



$$
\begin{align*}
\mathcal{L}(\lambda,y \mid x,\sigma) = \sigma f(x,y) + g(x,y) + \lambda_i h_i(y). 
\end{align*}
$$



为了保证约束问题集合的积极集对于扰动并不敏感，我们引入约束优化中常见的正则条件： 假设至少存在一个最优点 $y_{\sigma}^\ast(x) \in Y_{\sigma}^\ast(x)$，使得其对应的积极集中的所有 $\nabla_y h_i(y)$ 满足线性无关 (也称为LICQ正则条件)，并且存在一个对应的Lagrange乘子使得互补松弛条件满足。



## Proof: Part I

下面的记号采用 $w = (x,\sigma)$， $\theta = y$ , $p(w,\theta) = h_{\sigma}(x,y)$ 以及 $p^\ast(w) = l(x,\sigma)$. 假设代数性质 $\Theta = \{ h_i(\theta) \le 0\}$.



这些正则条件可以保证，


$$
\begin{align*}
\forall v \in {\rm Span} ( \nabla_{\theta w}^2 \,p(w,\theta) ) \Rightarrow 
\begin{bmatrix}
0 \\
v
\end{bmatrix}
\in {\rm Span} ( \nabla^2 \mathcal{L}(\lambda, \theta \mid w))
\end{align*}
$$


这个关系式的证明采用了反证法，假设存在 $v  \in {\rm Span} ( \nabla_{\theta w}^2 \,p(\theta,w) ) $ 然而 $(0,v) \notin {\rm Span} ( \nabla^2 \mathcal{L}(\lambda, \theta) \mid w)$， 那么我们可以将向量 $(0,v)$ 关于 Lagrange函数的Hessian矩阵的核空间和像空间进行分解 $(0,v) = v_{\parallel} + v_{\perp}$.   假设 $\Vert {\rm d}w \Vert \asymp \delta$ 满足 $v = \nabla_{\theta w}^2 \,p(\theta,w) {\rm d}w $ 以及 $ \Vert v \Vert \asymp \delta$.

在Proximal EB的假设下，最优解集合 $S(w) =  \arg \min_{\theta \in \Theta} p(w,\theta)$ 满足Lipschitz性质。此时存在 $\theta + {\rm d} \theta \in S(w + {\rm d} w)$ 满足 $\Vert {\rm d}  \theta \Vert = \mathcal{O}(\delta)$ 的时候，考虑其对应的Lagrange的一阶最优条件，成立如下的等式：



$$
\begin{align*}
\nabla \mathcal{L} (\lambda + {\rm d}\lambda , \theta + {\rm d} \theta \mid w + {\rm d} w) = 0.
\end{align*}
$$



相应的扰动项 ${\rm d} \lambda $ 也应该满足 $\Vert {\rm d} \lambda \Vert = \mathcal{O}(\delta)$. 这是由于 $ \Vert {\rm d} w \Vert = \mathcal{O}(\delta)$, $\Vert {\rm d} \theta \Vert = \mathcal{O}(\delta)$, 根据Lagrange乘子的有界性以及梯度的Lipschitz性质，我们知道



$$
\begin{align*}
\Vert \nabla \mathcal{L} (\lambda + {\rm d} \lambda ,\theta \mid w ) \Vert  = \mathcal{O}(\delta).
\end{align*}
$$


考虑积极集中的不等式约束，


$$
\begin{align*}
\Vert [\nabla h_1(\theta), \cdots, \nabla h_m(\theta)] {\rm d} \lambda \Vert  =  \Vert \nabla \mathcal{L} (\lambda + {\rm d} \lambda, \theta  \mid w) \Vert  = \mathcal{O}(\delta). 
\end{align*}
$$



根据 LICQ正则条件，我们知道 矩阵$[\nabla h_1(\theta), \cdots, \nabla h_m(\theta)]$ 的最小奇异值非零，利用SVD分解可以知道 $ \Vert {\rm d} \lambda  \Vert = \mathcal{O}(\delta) $.·



利用Taylor展开，我们知道


$$
\begin{align*}
 0  &= \nabla \mathcal{L}( \lambda + {\rm d} \lambda, \theta + {\rm d} \theta \mid w + {\rm d} w) - \nabla \mathcal{L}(\lambda , \theta \mid w) \\
&= \begin{pmatrix}
0 \\
\nabla_{\theta w}^2 p(w, \theta) {\rm d} w
\end{pmatrix} 
+ \nabla^2 \mathcal{L}( \lambda, \theta \mid w) 
\begin{pmatrix}
{\rm d} \lambda \\
{\rm d} \theta
\end{pmatrix}
+ o(\delta) \\
&=  v_{\parallel} + v_{\perp}  + \nabla^2 \mathcal{L}( \lambda, \theta \mid w) 
\begin{pmatrix}
{\rm d} \lambda \\
{\rm d} \theta
\end{pmatrix}
+ o(\delta) \\
\end{align*}
$$



由于 $v_{\perp} \notin {\rm Span} ( \nabla^2 \mathcal{L}(\lambda, \theta) \mid w)$, 上述的第三项无法消除 $v_{\perp}$ 这一项，因此右端项不为0, 那么就导出了矛盾。

 

## Proof: Part II



沿用上一个part的记号，我们知道 $v(w) = \min_{\theta \in \Theta} p(w,\theta)$. 我们考虑推广Danskin's 定理当 $\Theta^\ast(w)$ 满足 Lipschitz的情况.下面的证明可以在 [3] 中找到。

记 $\mathcal{I}_{\Theta}(\,\cdot\,)$ 为示性函数，根据集合 $\Theta$ 的凸性我们知道该示性函数为凸函数，利用该函数的帮助，我们知道



$$
\begin{align*}
v(w) = \min_{\theta \in \mathbb{R}^{d}} p(w,\theta) + \mathcal{I}_{\Theta}(\theta).
\end{align*}
$$



根据集合的Lipschitz性质，对于任意的 $\theta \in \Theta^\ast(w)$ 以及 ${\rm d} w $, 存在 $ \theta +t {\rm d} \theta \in \Theta^\ast(w + t{\rm d} w)$ 满足 $\Vert {\rm d} \theta \Vert \asymp  \Vert {\rm d} w \Vert $. 

根据一阶最优性条件，存在 

$$
\begin{align*}
\partial \mathcal{I}_{\Theta}(\theta)  \quad {\rm s.t.} \quad  \nabla_{\theta} p(w,\theta) + \partial \mathcal{I}_{\Theta}(\theta) = 0.
\end{align*}
$$

那么

$$
\begin{align*}
&\quad v(w +t{\rm d} w) - v(w) \\
&= p(w +t{\rm d} w,\theta + {\rm d}\theta) - p(w,\theta)+ \mathcal{I}_{\Theta}(\theta + t{\rm d} \theta) -  \mathcal{I}_{\Theta}(\theta) \\
& \ge \langle \nabla_w p(w,\theta), t{\rm d} w \rangle + \langle\nabla_{\theta} p(w,\theta), t{\rm d} \theta \rangle  + 
\langle \partial \mathcal{I}_{\Theta}(\theta), t{\rm d} \theta \rangle + o(t) \\
&= \langle \nabla_w p(w,\theta), t{\rm d} w \rangle + o(t).
\end{align*}
$$



同理，几乎对称地，成立


$$
\begin{align*}
&\quad v(w +t{\rm d} w) - v(w)  \\
&= p(w +t{\rm d} w,\theta + t{\rm d}\theta) - p(w,\theta)+ \mathcal{I}_{\Theta}(\theta + t{\rm d} \theta) -  \mathcal{I}_{\Theta}(\theta) \\ 
&\le   \langle \nabla_w p(w,\theta), t{\rm d} w \rangle +   \langle \nabla_{\theta} p(w +{\rm d} w, \theta +t{\rm d} \theta), t{\rm d} \theta \rangle + \langle \partial \mathcal{I}_{\Theta}(\theta + t{\rm d} \theta), t{\rm d} \theta \rangle + o(t)\\
&= \langle \nabla_w p(w,\theta), t{\rm d} w \rangle + o(t).
\end{align*}
$$



最后根据方向 ${\rm d}w$ 的任意性并且令 $t \rightarrow 0^+$ 就得到了如下推广后的Danskin's 定理



$$
\begin{align*}
\nabla v(w) = p(w, \theta), \quad \forall \theta \in \Theta^\ast(w).
\end{align*}
$$


## Proof: Part III



本节使用隐函数定理，给出 $\nabla^2 v(w)$ 的显示表达式，从而给出关于hyper-objective的梯度 $\nabla \varphi(x)$ 的显示表达式。



对于任意的扰动 ${\rm d} w$ , 根据集合的Lipschitz性质，对于任意的 $\theta \in \Theta^\ast(w)$ 以及 ${\rm d} w $, 存在 $ \theta + t{\rm d} \theta \in \Theta^\ast(w + t{\rm d} w)$ 满足 $\Vert {\rm d} \theta \Vert \asymp  \Vert {\rm d} w \Vert $.  

并且沿用之前的分析，我们知道对应的扰动后的Lagrange乘子也满足 $\lambda + t{\rm d} \lambda$ 也应该满足 $\Vert {\rm d} \lambda\Vert  \asymp  \Vert {\rm d} w \Vert$.



$$
\begin{align*}
&\quad \nabla v(w +t {\rm d} w) - \nabla v(w) \\
&= \nabla_w p(w +t {\rm d} w, \theta +t {\rm d} \theta ) - \nabla_w p(w,\theta)  \\ 
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w +  t  \nabla_{w \theta}^2p(w,\theta) {\rm d} \theta + o(t) \\
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w + 
[0 , \nabla_{w \theta}^2 p(w,\theta) ]
\begin{bmatrix}
t {\rm d} \lambda \\
t {\rm d } \theta
\end{bmatrix} + o(t).
\end{align*}
$$



令 $z = (\lambda, \theta)$, 那么根据Lagrange函数的一阶最优性条件 $\nabla \mathcal{L} (z \mid w) = 0$, 当 $ \nabla^2 \mathcal{L} (z \mid w)$ 可逆的时候，根据隐函数定理，有

$$
\begin{align*}
t{d} z = [\nabla^2 \mathcal{L} (z \mid w)]^{-1} 
\begin{bmatrix}
0 \\
\nabla_{\theta w}^2 p(w, \theta) t {\rm d} w
\end{bmatrix} + o(t).
\end{align*}
$$



此时使用双层优化问题的记号，回忆 $w = (\sigma,x)$ 就得到了



$$
\begin{align*}
&\quad  \frac{\partial^2 }{\partial \sigma\partial x } l(x,\sigma)  =  \frac{\partial^2 }{\partial x  \partial \sigma} l(x,\sigma) \\
&= \nabla_x f(x,y^\ast) - [0, \nabla_{xy}^2 h_{\sigma} (x,y^\ast) ] [\nabla^2 \mathcal{L}(\lambda^\ast,y^\ast \mid x, \sigma)]^{-1} 
\begin{bmatrix}
0 \\
\nabla_y f(x,y^\ast)
\end{bmatrix}, \quad \forall (\lambda^*, y^\ast) \in T(x,\sigma).
\end{align*}
$$



令 $\sigma  \rightarrow 0^+$ 就可以得到了 $\nabla \varphi(x)$ 的形式，该形式严格推广了强凸无约束情形下的 hyper-gradient的公式



$$
\begin{align*}
\nabla \varphi(x) = \nabla_x f(x,y^\ast(x)) - \nabla_{xy}^2 g(x,y^\ast(x)) [ \nabla_{yy}^2 g(x,y^\ast(x))]^{-1} \nabla_y f(x,y^\ast(x)).
\end{align*}
$$



下面考虑更一般的情形，也即当  $ \nabla^2 \mathcal{L} (z \mid w)$ 不一定可逆的时候。我们将证明，只要伪逆存在，那么可以使用伪逆推广上述的结论。

尽管类似的结论在一类特殊的PL函数上已经被相关工作证明 [6], 该文章所提供的证明以及技术都具有新的价值。

由于  $ \nabla^2 \mathcal{L} (z \mid w)$ 为对称矩阵，其伪逆可以被表达为 $U (U \nabla^2 \mathcal{L} U^\top)^{-1} U^\top $, 其中列正交矩阵 $U$ 对应于 $\nabla^2 \mathcal{L}$ 的非零特征值所张成的特征子空间。

令 $z_0$ 为 $z$ 到 ${\rm Ker}(\nabla^2 \mathcal{L})$  的投影，也即$z_0 = (I- UU^\top) z$. 我们知道 $z = U r + z_0$ 以及 $r = U^\top z$. 据此进一步定义 $\mathcal{L}_U(w,r) = \mathcal{L}(w, Ur + z_0)$.

注意到 $\nabla_r \mathcal{L}_U(w,r) = U^\top \nabla \mathcal{L}_z(w, z) = 0$, 根据该等式所定义的隐函数，有


$$
\begin{align*}
 {\rm d} r &=  - [ \nabla_{rr}^2 \mathcal{L}_U(w,r)]^{-1} \nabla_{wr}^2 \mathcal{L}_U(w,r) {\rm d}w + o(1) \\
&= - [U^\top \nabla_{zz}^2 \mathcal{L}(w,z) U]^{-1} U^\top \nabla_{zw}^2 \mathcal{L}(w,z) {\rm d}w + o(1)
\end{align*}
$$


利用该式子，并且注意到 Part I 中所证明的


$$
\begin{align*}
{\rm Span}(B) =  {\rm Span} \left( \begin{bmatrix}  0\\ \nabla_{\theta w}^2 p(w, \theta) \end{bmatrix} \right) \subseteq {\rm Span}(U).
\end{align*}
$$


因此我们可以用 $U$ 将矩阵 $B$ 线性表出，记 $B = U P$. 那么，


$$
\begin{align*}
&\quad \nabla v(w +t {\rm d} w) - \nabla v(w) \\
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w + 
B^\top
\begin{bmatrix}
t {\rm d} \lambda \\
t {\rm d } \theta
\end{bmatrix} + o(t) \\
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w + 
tP^\top U^\top  {\rm d}z + o(t) \\
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w + 
tP^\top {\rm d}r + o(t) \\
&= t\nabla_{ww}^2 p(w,\theta) {\rm d} w- t P^\top [U^\top \nabla_{zz}^2 \mathcal{L}(w,z) U]^{-1} U^\top \nabla_{zw}^2 \mathcal{L}(w,z) {\rm d}w + o(t) \\
&=t\nabla_{ww}^2 p(w,\theta) {\rm d} w- t P^\top U^\top U[U^\top \nabla_{zz}^2 \mathcal{L}(w,z) U]^{-1} U^\top \nabla_{zw}^2 \mathcal{L}(w,z) {\rm d}w + o(t) \\
&= t \nabla_{ww}^2 p(w, \theta) {\rm d} w - t  [0 ， \nabla_{w \theta}^2 p(w,\theta) ] [ \nabla \mathcal{L}(z \mid w)]^{\dagger} 
\begin{bmatrix}
0 \\
\nabla_{\theta w}^2 p(w, \theta) t {\rm d} w
\end{bmatrix} + o(t).
\end{align*}
$$




此时使用双层优化问题的记号，回忆 $w = (\sigma,x)$ 就得到了



$$
\begin{align*}
\nabla \varphi_{\sigma}(x) = \nabla_x f(x,y^\ast) - [0, \nabla_{xy}^2 h_{\sigma}(x,y^\ast)] [\nabla^2 \mathcal{L}(\lambda^\ast,y^\ast \mid x,\sigma)]^{\dagger} 
\begin{bmatrix}
0 \\
\nabla_y f(x,y^\ast)
\end{bmatrix}, \quad \forall(\lambda^\ast, y^\ast) \in T(x,\sigma).
\end{align*}
$$



这就完成了文章就主要的理论基础的证明，基于此文章提出了单循环的随机算法用于逼近 $\nabla \varphi(x)$ 来优化该问题。




## Reference



[1] On Penalty Methods for Nonconvex Bilevel Optimization and First-Order Stochastic Approximation. arXiv preprint. 203.

[2] Kwon J, Kwon D, Wright S, Nowak RD. A fully first-order method for  stochastic bilevel optimization. In ICML, 2023.

[3] Shen H, Chen T. On penalty-based bilevel gradient descent method. In ICML, 2023.

[4] Chen L, Xu J, Zhang J. Bilevel optimization without lower-level  strong convexity from the hyper-objective perspectuve. arXiv preprint 2023.

[5] Chen L, Ma Y, Zhang J. Near-Optimal Nonconvex-Strongly-Convex Bilevel Optimization with Fully First-Order Oracles. arXiv preprint 2023.

[6] Arbel M, Mairal J. Non-convex bilevel games with critical point  selection maps.  In NeurIPS, 2022.
