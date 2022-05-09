---
title: '最好的估计与最好的检验'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计
---



Cramér–Rao 下界与Neyman-Pearson引理简证。

<!--more-->



## Cramér–Rao Lower Bound



在实际中可以存在多个无偏估计量，Cramér–Rao 下界告诉我们哪一个统计量是最好的。Cramér–Rao 下界实际上是一个关于估计量的方差的不等式，达到不等式下界的估计量即为最小方差无偏估计量。该不等式实际上是关于Fisher信息的Cauthy不等式。

对于对数似然函数，成立如下性质，
$$
\begin{align*}
\mathbb{E}\left[ \frac{\partial \log f(X, \theta)}{\partial \theta}\right] = \int \frac{\partial f(X,\theta)}{\partial \theta}  \text{d}X = \frac{\partial}{\partial \theta} \int  f(X,\theta) \text{d}X = 0. 
\end{align*}
$$

以及，

$$
\begin{align*}
\mathcal{I}(\theta) = Var \left[ \frac{\partial \log f(X,\theta)}{\partial \theta}  \right]=\mathbb{E}\left[ \left(\frac{\partial \log f(X,\theta)}{\partial \theta}\right)^2 \right] = -\mathbb{E}\left[ \frac{\partial^2 \log f( X, \theta)}{\partial \theta^2}\right].
\end{align*}
$$

对于满足 $\mathbb{E}[T(X)] = g(\theta)$ 的无偏统计量，成立

$$
\begin{align*}
Cov\left[ T(X), \frac{\partial \log f(X,\theta)}{\partial  \theta} \right] = \int T(X) \frac{\partial f(X, \theta)}{\partial \theta} \text{d}X = \frac{\partial }{\partial \theta} \int T(X) f(X, \theta) \text{d}X = \frac{\partial g(\theta)}{\partial \theta}.
\end{align*}
$$

根据Cuthy不等式，

$$
\begin{align*}
Var[T(X)] Var \left[ \frac{\partial \log f(X,\theta)}{\partial \theta}  \right] \ge Cov\left[ T(X), \frac{\partial \log f(X,\theta)}{\partial  \theta} \right]
\end{align*}
$$

也即

$$
\begin{align*}
Var[T(X)] \ge  g'(\theta)/ \mathcal{I}(\theta).
\end{align*}
$$


## Neyman-Pearson Lemma



对于同一个假设检验问题，通常可以给出很多个检验统计量，而Neyman-Pearson告诉我们何者为最优的假设检验。

假设检验使用显著性水平 $\alpha$ 控制犯一类概率错误的概率，也即 $H_0$ 成立的前提下拒绝 $H_0$ 的概率，但却不能够控制犯二类错误的概率，也即 $H_1$ 成立下不拒绝 $H_0$ 的概率，而最优的假设检验应该是在显著性水平 $\alpha$ 的条件下犯第二类错误概率最小的假设检验。

Neyman-Pearson告诉我们最优的假设检验其实是如下的似然比检验，在如下的拒绝域下拒绝 $H_0$.
$$
\begin{align*}
R = \left\{\frac{f(X, \theta_1)}{f(X, \theta_0)} \ge \text{const} \right\}
\end{align*}
$$

下面进行证明，注意到犯一类错误的概率为，

$$
\begin{align*}
\mathbb{P}( \text{Type I}) = \int_R f(X, \theta_0 ) \text{d}X  = \alpha
\end{align*}
$$

而犯第二类错误的概率为，

$$
\begin{align*}
\mathbb{P}( \text{Type II}) = 1 - \int_R f(X, \theta_1) \text{d}X.
\end{align*}
$$
对于任意的其他假设检验，设其拒绝域为 $R'$, 可以验证如下不等式成立


$$
\begin{align*}
&\quad \mathbb{P}( \text{Type II} \vert R')  -  \mathbb{P}( \text{Type II} \vert R) \\
&= \int_{R} f(X, \theta_1) \text{d}X - \int_{R'} f(X, \theta_1) \text{d}X \\
&=  \int_{R- R'} f(X, \theta_1) \text{d}X - \int_{R'-R} f(X, \theta_1) \text{d}X \\
&\ge \text{const } \times  \left[ \int_{R- R'} f(X, \theta_0) \text{d}X - \int_{R'-R} f(X, \theta_0) \text{d}X \right] \\
& \ge \text{const } \times\left[ \int_{R} f(X, \theta_0) \text{d}X - \int_{R'} f(X, \theta_0) \text{d}X \right] \\
&= \text{const } \times [\mathbb{P}( \text{Type I} \vert R)  -  \mathbb{P}( \text{Type I} \vert R')] \\
&= \text{const }  \times [\alpha - \mathbb{P}( \text{Type I} \vert R')] \\
&\ge 0.
\end{align*}
$$
