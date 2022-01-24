---
title: 'Sub-Gradient Descent and Proximal Gradient Descent'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
---



次梯度方法和近端梯度方法的入门级介绍，其中次梯度方法多用于不可导的凸优化问题，而近端梯度方法多用于不可导的凸正则问题。

<!--more-->



主要参考自 [ECE236C - Optimization Methods for Large-Scale Systems](http://www.seas.ucla.edu/~vandenbe/ee236c.html)

## Sub Gradient Method

次梯度方法用于解决不可导的凸函数，次梯度定义为，


$$
\begin{align}
f(x) &\ge f(y) + g^T(x-y) 
\end{align}
$$

### Assumption

假设优化函数满足Lipschitz连续，


$$
\Vert f(x) - f(y) \Vert \le G \Vert x - y \Vert
$$


该性质等价于对次梯度的范数进行了限制，


$$
\begin{align}
\Vert g \Vert \le G
\end{align}
$$


充分性可以用Cathy-Schwarz不等式证明，必要性使用反证法即可，此处暂略.



基于上述假设提出次梯度方法，可以看作是梯度方法的自然拓展，

### Convergence

为了分析其收敛性，利用单步的迭代公式之后进行求和操作，


$$
\begin{align}
\Vert x_{k+1} - x_{\star} \Vert_2^2 &=\Vert x_{k} - \alpha_k g_k - x_{\star} \Vert_2^2 \\
&= \Vert x_k - x_{\star} \Vert_2^2 -2 \alpha_kg_k^T(x_k - x_{\star}) + \alpha_k^2 \Vert g_k \Vert_2^2 \\
& \le \Vert x_k - x_{\star} \Vert_2^2 +\alpha_k^2 \Vert g_k \Vert_2^2 + 2\alpha_k(f(x_{\star}) - f(x_k))\\ 
\Vert x_{k+1} - x_{\star} \Vert_2^2 - \Vert x_{k} - x_{\star} \Vert_2^2 &\le \alpha_k^2 \Vert g_k \Vert_2^2 + 2\alpha_k(f(x_{\star}) - f(x_k))\\ 
\Vert x_{k+1} - x_{\star} \Vert_2^2 - \Vert x_{0} - x_{\star} \Vert_2^2 &\le \sum_{i=1}^k \alpha_k^2 \Vert g_k \Vert_2^2 + 2\alpha_k(f(x_{\star}) - f(x_k))\\ 
f(x_k) - f(x_{\star}) &\le \frac{1}{2}(\frac{\sum_{i=1}^k \alpha_k^2 \Vert g_k \Vert_2^2}{\sum_{i=1}^k \alpha_k} + \frac{\Vert x_0 - x_{\star} \Vert_2^2}{\sum_{i=1}^k \alpha_k})
\end{align}
$$




假设我们知道初始点距离最优点的距离上界为$R$ , 并且设定学习率等恒定量，


$$
\begin{align}
f(x_k) - f(x_{\star}) &\le \min_{\alpha} \frac{1}{2} (\frac{k \alpha^2 G+R^2}{k \alpha})  \\
&= R^2 \sqrt{\frac{G}{k}} ,\text{With } \alpha = \frac{R^2}{\sqrt{Gk}}
\end{align}
$$


因此为了达到$\epsilon$-最优仅仅需要，


$$
k \ge O(\frac{1}{\epsilon^2})
$$


这正好达到了该问题的理论下界，证明需要构造一个对抗的反例说明任何该问题上的优化器都至少需要达到上面的界。





## Proximal Gradient Method



近端梯度下降方法也可以看作是梯度下降方法的自然延申，首先基于如下的观察，
$$
\begin{align}
x_{k+1} &= x_k - \alpha \nabla f(x) \\
x_{k+1} &= \min_{u} f(x) +  \nabla f(x)^T (u -x) + \frac{1}{2 \alpha} \Vert u-x \Vert_2^2 
\end{align}
$$


而如果优化目标加上一个不可导的项$h(x)$, 


$$
\begin{align}
x_{k+1} &= \min_{u} h(u) +f(x) +  \nabla f(x)^T (u -x) + \frac{1}{2 \alpha} \Vert u-x \Vert_2^2  \\
&= \min_{u} h(u) + \frac{1}{2 \alpha} \Vert u -x - \alpha \nabla f(x) \Vert_2^2 \\
&=\min_{u} \alpha h(u) + \frac{1}{2} \Vert u -x - \alpha \nabla f(x) \Vert_2^2 \\
&= \min_u \text{prox}_{\alpha h} (x - \alpha \nabla f(x)) \\
\text{With } \text{prox}_{h} &= \min_u h(u) + \frac{1}{2} \Vert u -x \Vert_2^2 
\end{align}
$$


上面定义的算子prox称为近端映射算子，其满足矫顽力等性质，


$$
\begin{align}
(\text{prox}_h(x) - \text{prox}_h(y))^T(x-y) \le \Vert  \text{prox}_h(x) - \text{prox}_h(y) \Vert_2^2 
\end{align}
$$


利用上面的结论加上Cauthy-Schwartz不等式也可以推出该算子是一个Lipschitz连续算子，


$$
\Vert \text{prox}_h(x) - \text{prox}_h(y) \Vert \le \Vert x - y \Vert
$$


### Assumption

本节关于近端梯度算法的分析中，假设函数具有如下性质，


$$
\begin{align}
g(y) &\ge g(x) + \nabla g(x)^T (y-x) + \frac{\mu}{2} \Vert y-x \Vert_2^2 \\
g(y) &\le g(x) + \nabla g(x)^T (y-x) + \frac{L}{2} \Vert y-x \Vert_2^2 \\
\end{align}
$$


定义近端梯度下降的单步下降，




$$
\begin{align}
x_{k+1} &=  x_k - \alpha G(x_k) \\
&= \min_u \text{prox}_{\alpha h} (x_k - \alpha \nabla g(x_k)) \\ 
G(x_k) &\in \nabla g(x_k) + \partial h(x_k - \alpha G(x_k)) 
\end{align}
$$


### Convergence

在该算法中，我们优化的目标仍然为$f(x) = g(x) + h(x) $，也即原本的可导的 $g(x)$ 加上不可导的惩罚项$h(x)$，例如可能是稀疏优化或者低秩优化中的内容，可以参见 [稀疏优化与低秩优化](https://truenobility303.github.io/Matrix-Complete/) 



我们首先关心近端梯度下降的方向是否使得函数值真的下降了，并且仍然简单地选取步长为$\frac{1}{L}$ , 可以发现的确为一个下降方向，



$$
\begin{align}
f(x_{k+1}) &= f(x_k - \alpha G_k) \\
&= g(x_k - \alpha G_k)+h(x_k - \alpha G_k) \\
&\le g(x_k)  - \frac{1}{L}\nabla g(x_k)^T G_k  +  \frac{1}{2L} \Vert G_k \Vert_2^2 + h(x_k - \frac{1}{L} G_k) ,\text{By Lipschitz}\\
&\le  g(x_k)  - \frac{1}{L}\nabla g(x_k)^T G_k  +  \frac{1}{2L} \Vert G_k \Vert_2^2 + h(x_k) - \frac{1}{L}(G_k - \nabla g(x_k))^TG_k ,\text{By Sub Gradient} \\
&= g(x_k) +h(x_k) - \frac{1}{2L} \Vert G_k \Vert_2^2 \\
&= f(x_k) -\frac{1}{2L} \Vert G_k \Vert_2^2 
\end{align}
$$



其次我们关心距离最优值的距离，最后求和并且利用函数值的单调性质，


$$
\begin{align}
f(x_{k+1}) &= f(x_k - \alpha G_k) \\
&= g(x_k - \alpha G_k)+h(x_k - \alpha G_k) \\
&\le g(x_k)  - \frac{1}{L}\nabla g(x_k)^T G_k  +  \frac{1}{2L} \Vert G_k \Vert_2^2 + h(x_k - \frac{1}{L} G_k) ,\text{By Lipschitz}\\
&\le  g(x_k)  - \frac{1}{L}\nabla g(x_k)^T G_k  +  \frac{1}{2L} \Vert G_k \Vert_2^2 + h(x_{\star}) - (G_k - \nabla g(x_k))^T(x_{\star} - x_k - \frac{1}{L} G_k) ,\text{By Sub Gradient} \\
&\le  g(x_{\star}) -  \nabla g(x_k) ^T(x_{\star} - x_k) - \frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2 - \frac{1}{L}\nabla g(x_k)^T G_k  +  \frac{1}{2L} \Vert G_k \Vert_2^2 + h(x_{\star}) - (G_k - \nabla g(x_k))^T(x_{\star} - x_k + \frac{1}{L} G_k) ,\text{By Stronly Convex} \\
&= f(x_{\star} )- G_k ^T(x_{\star} - x_k) - \frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2  -  \frac{1}{2L} \Vert G_k \Vert_2^2  \\ 
f(x_{k+1}) - f(x_{\star}) &\le - G_k ^T(x_{\star} - x_k) - \frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2  -  \frac{1}{2L} \Vert G_k \Vert_2^2 \\
&= \frac{L}{2}(\Vert x_{\star} - x_k \Vert_2^2- \Vert x_{\star}- x_k + \frac{1}{L} G_k \Vert_2^2) -\frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2 \\
&=\frac{L}{2}(\Vert x_{\star} - x_k \Vert_2^2- \Vert x_{\star}- x_{k+1} \Vert_2^2) -\frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2 \\
f(x_{k+1}) - f(x_{\star}) &\le \frac{1}{k} \sum_{i=1}^k f(x_{i+1}) - f(x_i) \\
&\le \frac{L}{2k}(\Vert x_{\star} - x_0 \Vert_2^2- \Vert x_{\star}- x_{k+1} \Vert_2^2) -\frac{\mu}{2k} \Vert x_{\star} - x_0 \Vert_2^2 \\
&\le \frac{L-\mu}{2k} \Vert x_{\star} - x_0 \Vert_2^2
\end{align}
$$


算法在$O(\frac{1}{\epsilon})$的时间内给出了$\epsilon$-最优解，再考虑距离最优点的距离是线性收敛的，

$$
\begin{align}
0 &\le \frac{L}{2}(\Vert x_{\star} - x_k \Vert_2^2- \Vert x_{\star}- x_{k+1} \Vert_2^2) -\frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2 \\
\Vert x_{\star}- x_{k+1} \Vert_2^2 &\le  \frac{L- \mu}{L} \Vert x_{\star}- x_{k} \Vert_2^2
\end{align}
$$

