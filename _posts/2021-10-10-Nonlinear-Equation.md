---
title:'非线性方程组的解法'
toc: true
excerpt_separator: <!--more-->
tags:
  - 数值算法
---

本文包括了二分法及其变种，不动点迭代法，和基于不动点迭代法的Newton迭代法，以及Secant方法、Broyden方法等经典的伪Newton方法。

<!--more-->

线性方程的解法通常基于高斯消元或者称为LU分解，本文关注于实际中更常见的非线性方程组。

## Bisection Method

### Naive Bisection

非线性方程组的解法本质为求根问题，也即寻找 $x$满足$f(x) = 0$.

二分法是最简单的求根方法，本质上利用的是零点存在性定理。

根据闭区间套定理，二分法的收敛性是显然的，且由于每次二分中点，设定数值精度$\tau$, 假定初始区间为$[a,b]$,二分法可以保证在$\log\frac{b-a}{\tau}$的步数内收敛。



### False Position Bisection

如果每次不是二分其中点，而是采用线性插值点作为新的二分点，将得到一种二分法的变种。

也即，相较于原来的方法，二分$c = \frac{a+b}{2}$。在变种的方法中，二分的是，$c = \frac{a f(b) - b f(a)}{f(b)-f(a)}$

上式$c$可以根据三点共线的条件解得，



$$
det 
\begin{pmatrix}
a  & f(a) & 1 \\
b  & f(b) & 1 \\
c & 0 & 1
\end{pmatrix}
= 0
$$


上述方法基于的思想在于，若在$[a,b]$内$f(x)$为线性函数，则上述方法找到的$c$恰好是线性函数的根。而当$[a,b]$区间长度很小的时候，$f(x)$也是接近于线性的。

上述方法同样也是保证收敛性的，但却不能保证收敛步数，在实际中可能慢于或者快于普通的二分法。



## Fixed Point Iteration

不动点迭代法是一种经典的解非线性方程组的方法，对于线性方程组不动点迭代法实际上为Jacobi迭代法。

不同于直接寻找$f(x)=0$的根，不动点迭代法将等式变形为，寻找$g(x)=x$的根。

并且基于不断将$g(x)$作用在原本的$x$上面进行迭代，


$$
x_{k+1} = g(x_k)
$$


简单的转换方法是，令$g(x) = f(x) +x$，但对于同个问题往往有多种转化方法，不同方法的效果取决于函数$g(x)$的性质。

下面根据$g(x)$的不同性质讨论其收敛性。



### Global Linear Coverage

当$ \Vert g'(x) \Vert < 1$的时候，不动点迭代法保证全局收敛性。

假设理论解为$x_{\star}$,其满足$x_{\star} = g(x_{\star})$，且在$x_k$的领域中，存在上界$L < 1$, $\Vert g'(x) \Vert \le L$,根据 Lagrange中值定理，


$$
\Vert x_{k+1} - x_{\star} \Vert = \Vert g(x_{k}) - g(x_{\star}) \Vert \le L \Vert x_k - x_{\star} \Vert
$$


由于$L <1 $，此时不动点迭代法收敛。

### Local Quadratic Coverage

当$g(x)$满足一些额外条件时，通常可以在局部得到更高阶的收敛性，例如:

当$g'(x_{\star}) = 0$的时候，不动点迭代法满足局部二次收敛（Quadratic Coverage）,证明在$x_{\star}$的领域使用Taylor展开即可，



$$
\Vert x_{k+1} - x_{\star} \Vert =  \Vert g(x_{k}) - g(x_{\star}) \Vert = \Vert \frac{g''(x_{\star})}{2}  (x_{k} - x_{\star})\Vert \le C\Vert x_{k} - x_{\star} \Vert^2,\exists C
$$



## Newton's Method

### Basic Newton Method

Newton法基于$x_k$的领域的Taylor展开，


$$
f(x_{k+1}) = f(x_k) + f'(x_k)(x_{k+1}-x_k) + o(\Vert x_{k+1} - x_k \Vert^2)
$$


在$x_k$的领域中，我们忽略二阶无穷小量，也即用切线近似函数。为了寻找到根，我们希望$f(x_{k+1})=0$，因此令，


$$
0 =  f(x_k) + f'(x_k)(x_{k+1}-x_k)
$$


从而得到Newton法的迭代公式，


$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$


Newton法也可以看作一种不动点迭代法:


$$
\begin{align}
\text{Let } & g(x) = x - \frac{f(x)}{f'(x)} \\
\text{Then } & g(x_{\star}) =0 \\
&g'(x_{\star}) = 1- \frac{(f'(x_{\star}))^2 - f(x_{\star}) f''(x_{\star})}{(f'(x_{\star}))^2} = 0
\end{align}
$$


因此，根据不动点迭代法的收敛条件，Newton迭代法也满足局部二次收敛。

---

Newton法有很多变种，比如在更新的时候加上系数$\lambda_k$作为阻尼因子也即带阻尼的牛顿法。



$$
x_{k+1} = x_k - \lambda_k \frac{f(x_k)}{f'(x_k)}
$$



如果对于一阶导数（Jacobi矩阵）采用某些近似方法，则称为伪牛顿法（Quasi-Newton Method），本节主要考虑几个经典的方法，并且证明其一些理论性质。



### Secant Method

Secant方法的核心是将Newton法中的一阶微分用一阶差分代替：



$$
\begin{align}
\text{Since } f'(x_k) &\approx \frac{f(x_k)-f(x_{k-1})}{x_k - x_{k-1}} \\
\text{Let } x_{k+1}& = x_k - \frac{f(x_k)(x_k - x_{k-1})}{f(x_k)-f(x_{k-1})} \\
\end{align}
$$



可以证明采用Secant方法，其收敛阶是黄金分割比例$\frac{1+\sqrt{5}}{2}$.

证明用到了Fibonacci数列形式的归纳递推，以及利用了Newton插值。

关于Newton插值，可以参见 [插值](https://truenobility303.github.io/Interpolation/)

利用Newton插值，给定$x_{k},x_{k-1}$，可以对任意点$x_{\star}$进行插值, 下式的$f[x_1,x_2,..,x_n]$表示Newton插值中的$n$阶差分。


$$
0 = f(x_{\star}) = f(x_{k}) + f[x_{k},x_{k-1}] (x_{\star}- x_k) + f[x_k,x_{k-1},x_{\star}] (x_\star - x_k) (x_{\star} - x_{k-1})
$$


利用差分的符号重写Secant法中的迭代公式，


$$
0 = f(x_k) + f[x_k,x_{k-1}] (x_{k+1}-x_k)
$$


将两式相减可以得到，


$$
x_{\star} - x_{k+1} = -\frac{f[x_k,x_{k-1},x_{\star}]}{f[x_k,x_{k-1}]} (x_{\star} - x_{k})(x_{\star}-x_{k-1})
$$


根据中值定理，并且假设在$x_{\star}$的领域内其一阶和二阶导数均有界，


$$
\Vert x_{\star} - x_{k+1} \Vert \le M \Vert x_{\star} - x_{k} \Vert \Vert x_{\star}-x_{k-1} \Vert, \exists M
$$


上式本质上和Fibonacci数列$F_k$非常相关，可以通过归纳法证明，归纳细节是显然的，只要注意参数的取值（包括归纳假设的满足等）


$$
\Vert  x_{\star} - x_k \Vert \le C \Vert x_{\star} - x_0 \Vert^{F_k}, \exists C
$$


再根据$F_k$的通项公式，则可以得到Secant方法的收敛阶是黄金分割比例$\frac{1+\sqrt{5}}{2}$.



### Good Broyden‘s Method

Broyden方法的思想是对Jacobi矩阵进行最小代价的秩一修正，使得Secant方法中的切线近似在修正后得到满足，


$$
\begin{align}
& J_k = J_{k-1} + \Delta J\\
& J_k \Delta x_k = \Delta f_k \\
\text{s.t. } & rank(\Delta J) = 1, \min \Vert \Delta J \Vert_F
\end{align}
$$


令$r = \Delta f_k - \Delta J_{k-1} \Delta x_k$, 则$\Delta J \Delta x = r$.

由于$rank(\Delta J)=1$, $\exists y, \Delta J = r y^T, y^T \Delta x = 1$

根据Frobenius范数和二范数的关系，有$ \Vert \Delta  J \Vert_F = \Vert r \Vert_2 \Vert y \Vert_2$

也即要最小化$\Vert y \Vert_2$, 又根据Cauthy不等式，$\Vert y \Vert_2 \Vert \Delta x \Vert_ 2 \ge 1$,因此$\Vert y \Vert_2 \ge \frac{1}{\Vert \Delta x \Vert_2}$

当且仅当$y = \alpha \Delta x$ 也即共线的时候取等，此时$\alpha = \frac{1}{\Delta x^T \Delta x}$

得到最终Jacobi矩阵的更新公式，


$$
J_k = J_{k-1} + (\Delta f_k - \Delta J_{k-1} \Delta x_k) \frac{\Delta x_k^T}{\Delta x_k^T \Delta x_k}
$$


由于Newton法中需要计算Jacopbi矩阵的逆矩阵，对于秩一修正的矩阵求逆，使用Shermann–Morrison–Woodbury公式，




$$
\begin{align}

J_k^{-1} &= (J_{k-1} + (\Delta f_k - \Delta J_{k-1} \Delta x_k) \frac{\Delta x_k^T}{\Delta x_k^T \Delta x_k})^{-1} \\
&= J_{k-1}^{-1}- \frac{J_{k-1}^{-1} (\Delta f_k - \Delta J_{k-1} \Delta x_k)  \Delta x_k^T J_{k-1}^{-1}}{\Delta x_k^T \Delta x_k+ \Delta x_k^T J_{k-1}^{-1}(\Delta f_k - \Delta J_{k-1} \Delta x_k)} \\
&= J_{k-1}^{-1} + \frac{(\Delta x_k - J_{k-1}^{-1} \Delta f_k) \Delta x_k^T J_{k-1}^{-1}}{\Delta x_k^T J_{k-1}^{-1}\Delta f_k}
\end{align}
$$


综上，得到了Good Broyden方法的迭代公式，


$$
\begin{align}

x_{k+1} &= x_k - J_k^{-1} f(x_k) \\
\text{With }J_k^{-1} &= J_{k-1}^{-1} + \frac{(\Delta x_k - J_{k-1}^{-1} \Delta f_k) \Delta x_k^T J_{k-1}^{-1}}{\Delta x_k^T J_{k-1}^{-1}\Delta f_k} \\
\Delta x_k & = x_k - x_{k-1}, \Delta f_k = f(x_k) - f(x_{k-1}) \\

\end{align}
$$


### Bad Broyden's Method

Bad Broyden方法不同于Good Broyden方法先对$J$做秩一修正，再用Shermann–Morrison–Woodbury公式求逆，Bad Broyden方法直接对$J^{-1}$做秩一修正。(这里的Bad 和Good其实只是方法名字的区分，并不代表实际效果的好坏)

也即满足秩一修正条件为，


$$
\begin{align}
& J_k^{-1} = J_{k-1}^{-1} + \Delta J\\
& J_k \Delta x_k = \Delta f_k \\
\text{s.t. } & rank(\Delta J) = 1, \min \Vert \Delta J \Vert_F
\end{align}
$$


此时定义，$r_k = \Delta x_k - J_{k-1}^{-1} \Delta f_k$,类似Good Broyden方法中的推导，$\Delta J \Delta f_k = r_k \\$。

再根据最小化$\Vert \Delta J \Vert_F $的条件，同样类似地可以得到，$\Delta J = \frac{r_k \Delta f_k^T}{\Delta f_k^T \Delta f_k}$

综上得到，最终迭代公式为，


$$
\begin{align}
x_{k+1} &= x_k - J_k^{-1} f(x_k) \\
\text{With }J_k^{-1} &= J_{k-1}^{-1} + (\Delta x_k - J_{k-1}^{-1} \Delta f_k) \frac{\Delta f_k^T}{\Delta f_k^T \Delta f_k} \\
\Delta x_k & = x_k - x_{k-1}, \Delta f_k = f(x_k) - f(x_{k-1}) \\
\end{align}
$$




