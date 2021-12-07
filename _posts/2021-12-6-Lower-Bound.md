---
title: '光滑-凸函数和光滑-强凸函数的复杂度下界'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
---


证明了对于连续可微的光滑-凸函数和光滑-强凸函数这两大类函数上一阶优化算法的复杂度下界，据此可以解释为什么Nesterov加速方法被称为最优算法。

<!--more-->


## Lower Bound for $\mathcal{F}_{L}$

通过构造一个特殊的函数，可以得到在一阶优化问题，优化目标为凸函数，并且满足$L$-光滑性质下算法的下界，也即存在一个函数使得任何算法都至少需要在该时间内进行优化。



定义一个特殊的三对角矩阵$A_{k \times k}$ ,


$$
\begin{align}
A = 
\begin{pmatrix}
2 & -1 & 0 & 0 &... & \\
-1 & 2 & -1 & 0 &... & \\
0 & -1 & 2 &-1 & ... & \\
0 &0 & -1 &2 & ... & \\
... &...&...&...&...
\end{pmatrix}
\end{align}
$$


在该矩阵的基础上可以定义函数，


$$
f_k(x) = \frac{L}{8} [x^TA x -e_1^Tx] ,\text{With } e_1 = [1,0,0,...,0]^T
$$


容易验证下面的式子成立


$$
\begin{align}
\nabla^2 f(x) &= \frac{L}{4}A\succeq 0 \\
L- \nabla^2 f(x) &= \frac{L}{4}(4I-A)\succeq 0
\end{align}
$$


结论是显然的，只需要使用对角占优矩阵一定是半正定阵的结论即可。



该函数的最优值也具有显式解，


$$
\begin{align}
\nabla f(x_{\star}) &= \frac{L}{2} (Ax_{\star} - e_1) = 0 \\
x_{\star}^{(i)} &= k x_{\star}^{(k)} \\
x_{\star}^{(i)} &= 1- \frac{i}{k+1} \\
f_{\star} &= f(x_{\star}) = \frac{L}{8} e_1^Tx_{\star} = \frac{L}{8}(1-\frac{1}{k+1})=   \frac{L}{8}\frac{k}{k+1}  
\end{align}
$$


假设迭代算法的初始点为原点$x_0 = 0$, 由于 $A$ 为三对角矩阵，我们可以证明对于任意使用一阶梯度优化的算法，其第 $k$ 步迭代到的位置 $x_k$ 都位于下面的子空间内，


$$
\begin{align}
x_k &\in \text{span} (\nabla f(x_0), \nabla f(x_1), ... , \nabla f(x_{k-1})) \\
&= \text{span} (e_1,e_2,...,e_k)
\end{align}
$$


也即 $x_k$ 每次仅仅在一个新的一个维度张成的子空间内，证明只需要使用数学归纳法，归纳假设由 $x_0 = 0, \nabla f(x_0 ) \in \text{span}(e_1)$ 得.



利用上述这个重要的性质，定义优化的目标为 $f_{2k+1}$ 可以得到该问题的下界。


$$
\begin{align}
\frac{f_{2k+1}(x_k) - f_{2k+1}(x_{\star})}{\Vert x_0 - x_{\star} \Vert^2} & = \frac{f_{2k+1}(x_k) - f_{2k+1}(x_{\star})}{\Vert x_{\star} \Vert^2} \\
&= \frac{f_{k}(x_k) - f_{2k+1}^{\star}}{\Vert x_{\star} \Vert^2} \\
&\ge \frac{f_{k}^{\star} - f_{2k+1}^{\star}}{\Vert x_{\star} \Vert^2} \\
&= \frac{\frac{L}{8}(\frac{1}{k+1} - \frac{1}{2k+2})}{ \Vert x_{\star} \Vert^2} \\
&= \frac{L}{16(k+1)\Vert x_{\star} \Vert^2} \\
&=  \frac{L}{16(k+1) \sum_{i=1}^{2k+1} (1-\frac{i}{2k+2})^2} \\
&= \frac{L}{16(k+1)((2k+1) - \frac{1}{k+1}\sum_{i=1}^{2k+1} i+  \frac{1}{4(k+1)^2}\sum_{i=1}^{2k+1} i^2} \\
&= \frac{3L}{4(2k+1)(4k+3)} \\ 
&\ge  \frac{3L}{32(k+1)^2}
\end{align}
$$



利用级数放缩的技巧，还可以证明，对于距离最优点的距离满足，


$$
\begin{align}
\Vert x_k - x_{\star} \Vert^2 &\ge \sum_{k+1}^{2k+1} (x_{2k+1}^{(i)})^2 \\
&=\sum_{i=k+1}^{2k+1} (1-\frac{i}{2k+2})^2 \\
&=k+1 -\frac{1}{k+2} \sum_{i=k+1}^{2k+1} i + \frac{1}{4(k+1)^2} \sum_{i=k+1}^{2k+1} i^2 \\
&= (k+1) - \frac{3k+2}{2} + \frac{(2k+1)(7k+6)}{24(k+1)} \\
&= \frac{2k^2+7k+6}{24(k+1) } \\

\end{align}
$$


并且根据上面的放缩，


$$
\begin{align}
\Vert x_0 - x_{\star} \Vert^2 &= \Vert x_{\star} \Vert^2 \\
&= \sum_{i=1}^{2k+1} (1- \frac{i}{2k+2})^2 \\
&= \frac{1}{4(k+1)^2} \sum_{i=1}^{2k+1} i^2 \\
&= \frac{(2k+1)(2k+2)(4k+3)}{24(k+1)^2} \\
& \le \frac{16(k+1)^3}{24(k+1)^2} \\
&= \frac{2}{3} (k+1)
\end{align}
$$
联合两式就得到了，


$$
\begin{align}
\frac{\Vert x_k - x_{\star} \Vert^2}{\Vert x_0 - x_{\star} \Vert^2} &\ge \frac{2k^2+7k+6}{16(k+1)^2 } \ge \frac{2k^2+4k+4}{16(k+1)^2 } = \frac{1}{8} 
\end{align}
$$


因此利用该构造的函数得到的优化复杂度下界为，


$$
\begin{align}
f(x_k) - f(x_{\star}) &\ge \frac{3L}{32(k+1)^2} \Vert x_0 - x_{\star} \Vert^2 \\
\Vert x_k - x_0 \Vert^2 &\ge \frac{1}{8} \Vert x_0 - x_{\star} \Vert^2
\end{align}
$$


## Lower Bound for $\mathcal{F}_{L,\mu}$



我们关心如果优化的函数类不仅满足$L$-光滑性质，而且满足$\mu$- 强凸的时候，复杂度的下界，该结果可以利用上面的函数进行微小的改动得到。



定义如下的函数 $f(x)$ 并且验证其性质，


$$
\begin{align}
f(x) &= \frac{L-\mu}{8}(x^TAx - e_1^Tx) + \frac{\mu}{2} \Vert x \Vert^2 \\
\nabla^2 f(x) &= \frac{L-\mu}{4} A + \mu \\
\mu &\preceq \nabla^2 f(x) \preceq L
\end{align}
$$


完全类似地，假设迭代算法的初始点为原点$x_0 = 0$, 由于 $A$ 为三对角矩阵，我们可以证明对于任意使用一阶梯度优化的算法，其第 $k$ 步迭代到的位置 $x_k$ 都位于下面的子空间内，

$$
\begin{align}
x_k &\in \text{span} (\nabla f(x_0), \nabla f(x_1), ... , \nabla f(x_{k-1})) \\
&= \text{span} (e_1,e_2,...,e_k)
\end{align}
$$


也即 $x_k$ 每次仅仅在一个新的一个维度张成的子空间内，证明只需要使用数学归纳法，归纳假设由 $x_0 = 0, \nabla f(x_0 ) \in \text{span}(e_1)$ 得.



我们计算该函数的最优点，可以得到下述方程


$$
\begin{align}
\nabla f(x_{\star}) &= \frac{L-\mu}{4} (Ax_{\star}-e_1x_{\star}) + \mu x_{\star} = 0 \\
e_1 &=(\frac{A}{4} + \frac{\mu}{L- \mu} ) x_{\star} \\
0 &=x_{\star}^{(k+1) } +2 \frac{L+\mu}{L-\mu} x_{\star}^{(k)} + x_{\star}^{(k-1) }  
\end{align}
$$


根据递推公式可以得到其解，


$$
x_{\star}^{(k)} = q^k ,q=\frac{\sqrt L+\sqrt \mu}{\sqrt L- \sqrt \mu}
$$




利用上述条件可以得到该问题的复杂度下界，


$$
\begin{align}
\frac{\Vert x_k - x_{\star \Vert^2}}{ \Vert x_0 -x_{\star} \Vert^2} &= \frac{\Vert x_k - x_{\star \Vert^2}}{ \Vert x_{\star} \Vert^2} \ge \frac{\sum_{i={k+1}}^{\infty} (x_{\star}^{(i)})^2}{\sum_{i=1}^{\infty} (x_{\star}^{(i)})^2} = q^{-2k} \\
\Vert x_k - x_{\star} \Vert^2 &\le \frac{1}{q^{2k}}  \Vert x_0 -x_{\star} \Vert^2 \\
f(x_k) - f(x_\star) &\ge \frac{\mu}{2} \Vert x_k - x_{\star}\Vert^2 \ge \frac{\mu}{2} \frac{1}{q^{2k}} \Vert x_0 -x_{\star} \Vert^2
\end{align} 
$$


## Optimal Method

通过比较优化算法距离上述复杂度下界的距离，我们可以衡量一个优化算法的好坏，我们希望找到一个能够达到上述复杂度下界的优化算法，该算法就是该问题的最优算法。

对于非强凸的光滑函数，根据 [梯度法](https://truenobility303.github.io/CG/) 中得到的结论，我们知道其为了达到 $\epsilon$- 最优解，需要的迭代次数为 $O(\frac{1}{\epsilon})$ 次迭代，而距离复杂度下界推导出来的 $O(\frac{1}{\sqrt \epsilon})$ 次迭代仍然有一定的差距，因此梯度法并非该问题的最优算法。

但对于 [FISTA](https://truenobility303.github.io/FISTA/) 算法，其神奇的迭代形式正好使得其只需要  $O(\frac{1}{\sqrt \epsilon})$ 次迭代就可以达到  $\epsilon$- 最优解，根据本文的结论，这个界已经不能再被优化了，因此 FISTA算法是该问题的最优算法。

类似地，可以证明，[Nesterov 加速方法](https://truenobility303.github.io/Nesterov-Acceleration/) 可以达到强凸-光滑函数类的复杂度下界，但其推导过程稍显繁琐，此处直接给出该结论，对该结论的证明感兴趣的读者可以参见 Yurii Nesterov的Lectures on Convex Optimization一书。



