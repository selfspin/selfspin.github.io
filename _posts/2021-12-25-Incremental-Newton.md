---
title: '增量式牛顿算法'
toc: true
excerpt_separator: <!--more-->
tags
  - 优化
---



论文阅读笔记：[A Newton-type Incremental Method with a Superlinear Convergence Rate](http://opt-ml.org/oldopt/papers/OPT2015_paper_16.pdf)



<!--more-->





## Method Overview



论文关注于以下强凸函数的优化问题，


$$
\min f(x): = \frac{1}{n} \sum_{i=1}^n f_i(x) + \frac{\mu}{2} \Vert x \Vert^2
$$


算法类似于SGD, 且每次仅仅使用一个样本，用样本增量式地维护一阶和二阶信息的更新，并且使用函数的凸二次近似进行优化，


$$
\begin{align}
\min_{x_{k+1}} m_i^{(k)}(x_k):&= \frac{1}{n} [\sum_{i=1}^n f_i(v_k^i) + \nabla f_i(v_k^i)(x_k- v_k^i) + \frac{1}{2} (x_k- v_k^i) \nabla^2 f_i(v_k^i)(x_k- v_k^i)] + \frac{\mu}{2} \Vert x_k \Vert^2\\
x_{k+1} &=   (H_k + \mu I)^{-1} (u_k - g_k) \\
\text{With } H_k:&= \frac{1}{n} \sum_{i=1}^n \nabla^2 f_i(v_k^i), u_k := \frac{1}{n} \sum_{i=1}^n \nabla^2 f_i(v_k^i)v_k^i ,g_k :=\frac{1}{n} \sum_{i=1}^n \nabla f_i(v_k^i)
\end{align}
$$

算法的独特之处在于其每次仅仅轮换地更新一个样本的信息，也即选择一个样本 $v_{k+1}^i$ 变为 $x_{k+1}$ , 证明中需要算法更新顺序依照样本轮换更新，但实际中也可以采用随机选取样本进行更新的方式。



算法证明了其局部超线性收敛和全局线性收敛性质，下面我们逐次证明。



## Local SuperLinear Coverage

局部超线性收敛性的证明的核心是利用与增量式轮换更新相关的递归式数列估计收敛率。



需要假设如下的Hessian矩阵Lipschitz连续的条件，且初始点的位置距离最优值比较接近，


$$
\Vert \nabla^2 f_i(x) - \nabla^2 f_i(y) \Vert \le M \Vert x- y\Vert
$$


考虑算法距离最优点的距离，


$$
\begin{align} 
x_{k+1} - x_{\star} &=(H_k + \mu I)^{-1} (u_k - g_k) - x_{\star} \\
&= (H_k + \mu I)^{-1}(u_k - g_k - (H_k + \mu I)x_{\star}) \\
&= A_k^{-1} (u_k - g_k - (H_k + \mu I)x_{\star}) ,\text{Let } A_k = H_k + \mu I\\
&= A_k^{-1} ( \frac{1}{n} \sum_{i=1}^n \nabla^2 f_i(v_k^i)(v_k^i - x_{\star}) - \frac{1}{n} \sum_{i=1}^n(\nabla f_i(v_k^i) - \nabla f_i(x_{\star}))\\
&= \frac{1}{n}A_k^{-1} ( \sum_{i=1}^n \nabla^2 f_i(v_k^i)(v_k^i - x_{\star}) - \sum_{i=1}^n(\nabla f_i(v_k^i) - \nabla f_i(x_{\star}))\\
&= \frac{1}{n}A_k^{-1} \sum_{i=1}^n [\nabla^2 f_i(v_k^i)(v_k^i - x_{\star}) - \int_0^1 \nabla^2 f_i(v_k^i +t(v_k^i - x_{\star}))  (v_k^i -x_{\star})dt]\\
&=  \frac{1}{n}A_k^{-1} \sum_{i=1}^n \int_0^1[\nabla^2 f_i(v_k^i)(v_k^i - x_{\star}) -\nabla^2 f_i(v_k^i +t(v_k^i - x_{\star}))  (v_k^i -x_{\star})dt]\\
\Vert x_{k+1} - x_{\star} \Vert &\le \frac{M}{n\mu} \sum_{i=1}^n \int_0^1 \Vert v_k^i -x_{\star} \Vert^2 t dt \\
&= \frac{M}{2n\mu} \sum_{i=1}^n \Vert v_k^i - x_{\star} \Vert^2 \\
&=\frac{M}{2n\mu} \sum_{i=1}^n \Vert x_{k-i+1} - x_{\star} \Vert^2 \\
\tilde r_{k+1} &\le  \frac{M}{2n\mu} [\tilde r_k^2 + \tilde r_{k-1}^2 + ... + \tilde r_{k-n+1}^2 ] 
\end{align}
$$


为了对 $\tilde r_k$ 进行估计，我们定义其一个递归式上界序列 $r_k $，并且推导该序列的性质


$$
r_k := \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2], C= \frac{M}{2 \mu}
$$


利用最大值对其进行基本的估计，


$$
\begin{align}
r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\le C \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] \\
r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\ge \frac{C}{n} \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2]
\end{align}
$$


当初始值足够小的时候，也即满足算法的局部性条件的时候，可以证明序列的元素都将足够小，可以通过归纳法证明，


$$
\begin{align}
\text{If } \max[r_0,r_1,...r_{n-1}] &\le \frac{1}{C \sqrt n} \\
\text{Then } r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\le C \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] \\ \\
&\le \frac{1}{C n} \le \frac{1}{C \sqrt n}
\end{align}
$$


下面证明一个辅助性的引理，说明了在局部性质的假设前提下，该序列的最大值也具有二次收敛性质，同样利用递归证明，


$$
\begin{align}
r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\le C \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] \\ 
r_{k+1} &= \frac{C}{n} [r_{k}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\
&\le C \max[r_k^2, r_{k-1}^2, ..., r_{k-n}^2] \\
&\le C\max [C \max[r_{k-1}^4 , r_{k-2}^4 , ... ,r_{k-n}^4] , r_{k-1}^2 , ... ,r_{k-n+1}^2] \\
&\le C\max [\max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] , r_{k-1}^2 , ... ,r_{k-n+1}^2] ,\text{By } r_k \le \frac{1}{C \sqrt n} \\
&= C \max[r_{k-1}^2 , ... ,r_{k-n+1}^2] \\
&\le C \max[r_{k-1}^2 , ... ,r_{k-n+1}^2, r_{k-n}^2] \\
r_{k+2} &=\frac{C}{n} [r_{k+1}^2 + r_{k}^2 + ... +r_{k-n+2}^2] \\
&\le ... \\
&\le  C \max[r_{k-1}^2 , ... ,r_{k-n+1}^2, r_{k-n}^2] \\
r_{k+n+1} &= ...
\end{align}
$$


合起来得到最终的引理为，


$$
\max[r_{k+n-1}, r_{k+n-2}, ...,r_k] \le C \max[r_{k-1}^2, r_{k-2}^2,... ,r_{k-n}^2]
$$


据此可以得到当 $k \ge 2n$ 的时候，序列满足单调性，


$$
\begin{align}
r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\le C \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] \\  
&\le  C ( C  \max [r_{k-n-1}^2 , r_{k-n-2}^2 , ... ,r_{k-2n}^2])^2 \\
&\le C^3 \max [r_{k-n-1}^4 , r_{k-n-2}^4 , ... ,r_{k-2n}^4] \\
&\le \frac{C}{n} \max [r_{k-n-1}^2 , r_{k-n-2}^2 , ... ,r_{k-2n}^2], \text{By } r_k \le \frac{1}{C \sqrt n} \\
&\le r_{k-n} \\
r_{k+1} &=\frac{C}{n} [r_{k}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\
&= r_{k} +\frac{C}{n}[r_k^2 - r_{k-n}^2] \\
&\le r_k
\end{align}
$$


如果进一步进行推导可以发现当 $k \ge 3n$ 的时候序列的二次收敛性，只需要利用单调性去掉 $\max$ 算子即可，


$$
\begin{align}
r_k &= \frac{C}{n} [r_{k-1}^2 + r_{k-2}^2 + ... +r_{k-n}^2] \\
&\le C \max[r_{k-1}^2 , r_{k-2}^2 , ... ,r_{k-n}^2] \\   
&\le C r_{k-n}^2
\end{align}
$$


据此可以得到当 $k \ge 3n$ 的时候序列线性收敛，




$$
\begin{align}
r_{k+1} &= \frac{C}{n} [r_{k}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\ 
&\le \frac{C}{n} [C^2r_{k-n}^4 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\ 
&\le \frac{C}{n} [\frac{1}{n}r_{k-n}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2], \text{By } r_k \le \frac{1}{C \sqrt n} \\
&= \frac{C}{n} [\frac{1}{n}r_{k-n}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\
&= \frac{C}{n} [\frac{1}{n}r_{k-n}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2 +r_{k-n}^2 - r_{k-n}^2] \\
&=r_k - \frac{C(n-1)}{n^2} r_{k-n}^2 \\
&\le r_k - \frac{n-1}{n^2} r_k,\text{By } r_k \le C r_{k-n}^2 \\
&=q r_k, \text{Let } q = 1- \frac{n-1}{n^2}
\end{align}
$$


但我们并不仅仅满足于此，实际上每过 $n$ 轮超线性系数 $q$ 都为以平方速度增长，


$$
\begin{align}
r_{k+1} &= \frac{C}{n} [r_{k}^2 + r_{k-1}^2 + ... +r_{k-n+1}^2] \\  
&\le \frac{q^2C}{n} [r_{k-1}^2+r_{k-2}^2 + ... +r_{k-n}^2] \\
&= q^2 r_k
\end{align}
$$


因此实际上该序列是超线性收敛的，据此我们成功地证明了算法的局部超线性收敛性，


$$
\begin{align}
\frac{r_{k+1}}{r_k} &\le q^{2^{[k /n]-3}} , k\ge 3n ,q<1\\
\lim_{k \rightarrow \infty} \frac{r_{k+1}}{r_k} &=0
\end{align}
$$


## Global Linear Coverage



本节证明算法的全局线性收敛性质



用到的假设是每个样本函数 $L$ -光滑的性质，


$$
\Vert \nabla f_i(x) - \nabla f_i(y) \Vert \le L \Vert x - y \Vert
$$

蕴含着总体函数 $L + \mu$ - 光滑的性质， 


$$
\Vert \nabla f(x) - \nabla f(y) \Vert  \le (L+ \mu )\Vert x- y \Vert
$$




证明的思想在于使用迭代方向之前的误差，

$$
\begin{align}
x_{k+1} &= x_k + \alpha p_k \\
&=x_k + \alpha q_k +\alpha e_k \\
\text{With } p_k &:=A_k^{-1} (\frac{1}{n} \sum_{i=1}^n \nabla^2 f_i(v_k^i)(v_k^i - x_k) -\frac{1}{n} \sum_{i=1}^n\nabla f_i(v_k^i) - \mu x_k ) \\
q_k&:= -A_k^{-1} \nabla f(x_k) = -A_k^{-1}(\frac{1}{n} \sum_{i=1}^n \nabla f_i(v_k^i)+\mu x_k)  \\
e_k &:= p_k -q_k
\end{align}
$$




并且试图利用梯度的范数作为上界进行控制，首先 $q_k$ 的范数是简单的，


$$
\begin{align}
\Vert q_k \Vert \le \Vert A_k^{-1} \Vert \Vert \nabla f(x_k) \Vert \le \frac{1}{\mu} \Vert \nabla f(x_k) \Vert
\end{align}
$$

更重要的是控制 $e_k$ 的范数，


$$
\begin{align}
e_k &= A_k^{-1} (\frac{1}{n} \sum_{i=1}^n \nabla^2 f_i(v_k^i)(v_k^i - x_k) -\frac{1}{n} \sum_{i=1}^n (\nabla f_i(v_k^i) -\nabla f_i(x_k)) \\
\Vert e_k \Vert &\le \frac{\Vert A_k^{-1} \Vert}{n} [\sum_{i=1}^n L \Vert v_k^i - x_k \Vert +\sum_{i=1}^nL \Vert v_k^i - x_k \Vert]\\ 
&\le  \frac{2L}{n \mu} \sum_{i=1}^n  \Vert v_k^i - x_k \Vert \\
&=\frac{2L}{n \mu} [\Vert x_{k-1} - x_k \Vert + \Vert x_{k-2} - x_k \Vert + ... +\Vert x_{k-n-1} - x_k \Vert ] \\
&\le \frac{2L}{\mu } \max_{k-n+1 \le j \le k-1} \Vert x_j - x_k \Vert \\
&\le \frac{2L}{\mu } \sum_{k-n+1 \le j \le k-1} \Vert x_{j+1} - x_{j} \Vert \\
&\le \frac{2L(n-1)}{\mu} \max_{k-n+1 \le j \le k-1} \Vert x_{j+1} - x_{j} \Vert  \\
\end{align}
$$


另一方面，



$$
\begin{align}
\Vert e_k \Vert &\le \frac{2L}{\mu } \max_{k-n+1 \le j \le k-1} \Vert x_j - x_k \Vert \\ 
&\le \frac{2L}{\mu } \max_{k-n+1 \le j \le k-1} [\Vert x_j - x_{\star} \Vert + \Vert x_k - x_{\star} \Vert] \\
&\le \frac{4L}{\mu } \max_{k-n+1 \le j \le k} \Vert x_j - x_{\star} \Vert \\
&\le \frac{4L}{\mu^2} \max_{k-n+1 \le j \le k} \Vert \nabla f(x_j)\Vert \\
\end{align}
$$



而根据迭代公式，


$$
\begin{align}
x_{k+1} -x_k  &= \alpha q_k +\alpha e_k \\
\Vert x_{k+1} -x_k \Vert &\le \alpha \Vert q_k \Vert + \alpha \Vert e_k \Vert \\
&\le \frac{\alpha}{\mu} \Vert \nabla f(x_k) \Vert+ \alpha \Vert e_k \Vert) \\
&\le  \frac{\alpha}{\mu} \Vert \nabla f(x_k) \Vert+\frac{4\alpha L}{\mu^2} \max_{k-n+1 \le j \le k} \Vert \nabla f(x_j)\Vert \\
\end{align}
$$
嵌入得到，


$$
\begin{align}
\Vert e_k \Vert &\le \frac{2L(n-1)}{\mu} \max_{k-n+1 \le j \le k-1} \Vert x_{j+1} - x_{j} \Vert  \\
&\le \frac{2L(n-1)}{\mu} \max_{k-n+1 \le j \le k-1}[ \frac{\alpha}{\mu} \Vert \nabla f(x_j) \Vert+\frac{4\alpha L}{\mu^2} \max_{j-n+1 \le m \le j} \Vert \nabla f(x_m)\Vert] \\
&\le \frac{2L(n-1)}{\mu} (\frac{\alpha}{\mu} + \frac{4\alpha L}{\mu^2}) \max_{k-2n+2\le j \le k-1} \Vert \nabla f(x_j ) \Vert \\
&= \frac{2\alpha (n-1)L(\mu+4L)}{\mu^3}  \max_{k-2n+2\le j \le k-1} \Vert \nabla f(x_j ) \Vert \\
\end{align}
$$


基于上述引理的准备，观察函数值的下降，




$$
\begin{align}
f(x_{k+1}) - f(x_k) &\le \nabla f(x_k)^T(x_{k+1} - x_k) + \frac{L}{2} \Vert x_{k+1} - x_k \Vert^2 \\
&= \alpha \nabla  f(x_k)^T(q_k+ e_k) + \frac{ \alpha^2 L}{2} \Vert q_k + e_k \Vert^2 \\
&\le -\alpha \nabla f(x_k) A_k^{-1} \nabla f(x_k) + \alpha \nabla f(x_k)^T e_k +\frac{ \alpha^2 L}{2} \Vert q_k + e_k \Vert^2 \\
&\le -\alpha L \Vert \nabla f(x_k) \Vert^2 +  \alpha \nabla f(x_k)^T e_k +\frac{ \alpha^2 L}{2} \Vert q_k + e_k \Vert^2 \\ 
&\le  -\alpha L \Vert \nabla f(x_k) \Vert^2 + \frac{\alpha^2 L}{2\mu^2} \Vert \nabla f(x_k) \Vert^2 + (\frac{\alpha^2 LC}{\mu}+ \frac{\alpha^2LC^2}{2}+ \frac{\alpha L}{\mu})  \max_{k-2n+2\le j \le k} \Vert \nabla f(x_j ) \Vert^2 \\
\text{Where } C &= \frac{2\alpha (n-1)L(\mu+4L)}{\mu^3}
\end{align}
$$


利用函数的性质有，


$$
2 \mu (f(x) - f(x_{\star})) \le \Vert f(x) \Vert^2  \le 2L (f(x) - f(x_{\star}))
$$


定义距离最优值的距离 $V_k := f(x_k ) - f(x_{\star})$ ,  结合上述两个式子可以得到，


$$
V_{k+1} \le p(\alpha ) V_k + q(\alpha) \max_{k-2n+2 \le j\le k} V_j
$$


最终只需要选取步长 $\alpha$ 满足 $p(\alpha)  + q(\alpha) < 1$ 就可以保证算法线性收敛，依赖于下面基于归纳法的结论，


$$
\begin{align}
\text{If } V_{k+1} &\le p(\alpha) V_k +q(\alpha) \max_{j \le k- d \le k} V_j \\
p(\alpha) + q(\alpha) &<1 \\
V_k &\le r^k, r= (p+q)^{\frac{1}{d+1}} \\
\text{Then} V_{k+1} &\le p(\alpha) V_k +q(\alpha) \max_{j \le k- d \le k} V_j \\
&\le p(\alpha ) r^k+ q(\alpha ) r^{k-d} \\
&=r^{k-d} (p(\alpha ) r^d +q(\alpha) ) \\
&\le r^{k-d} (p(\alpha )  +q(\alpha) ) \\
&=r^{k-d} r^{d+1} \\
&=r^{k+1}
\end{align}
$$




选取的步长显示表达式较为复杂，此处仅仅证明到这里为止。
