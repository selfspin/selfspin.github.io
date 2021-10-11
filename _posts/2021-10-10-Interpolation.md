---
title: '插值'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - 数值算法
---

多项式插值的基本内容，包括经典的Lagrange,Neville,Newton插值方法，以及多项式重构问题以及Chebyshev插值，Hermite插值于矩阵函数的关系，样条插值以及其性质。

<!--more-->



## Polynomial Interpolation

本节研究多项式插值问题，给定$n$个数据结点$(x_1,y_1),(x_2,y_2),...,(x_n,y_n)$，求一个$n-1$阶以内多项式，满足：



$$
p(x_i) = y_i
$$



### Uniqueness

上述问题解的唯一性可以简单证明。

一种证明方法是利用线性方程组的思想，将多项式插值问题看作：



$$
\begin{pmatrix}
1 & x_1 & ... & x_1^{n-1} \\
1 & x_2 & ... & x_2^{n-1} \\
...&...&...&...\\
1& x_n &... & x_n^{n-1}
\end{pmatrix}
\begin{pmatrix}
a_0 \\
a_1 \\
... \\
a_{n-1}
\end{pmatrix}
=
\begin{pmatrix}
y_1 \\
y_2 \\
... \\
y_{n} 
\end{pmatrix}
$$



其系数矩阵为Vandemonde行列式，记作$A$,则有：



$$
det(A) = \prod_{i <j} (x_j - x_i)
$$



由于$x_i \ne x_j$，因此$det(A) \ne 0$，因此上述线性方程有唯一解。

---

另一种证明思路是首先构造出一个满足条件的多项式$p_1(x)$，然后证明任意满足条件的多项式$p_2(x)$都与$p_1(x)$相等。

此时，$n-1$阶多项式$p_1(x) - p_2(x)$有$n$个相异的根$x_1,x_2,...,x_n$，该$n-1$阶多项式只能为0多项式，也即$p_1(x) = p_2(x)$.





### Lagrange‘s Method

Lagrange插值多项式是一种常见的构造方法。

其基于的思想也很简单，构造出一组多项式基$L_i(x)$，满足：



$$
\begin{align}
L_i(x_j) = 1 ,i=j \\
L_i(x_j) = 0,i \ne j
\end{align}
$$

$L_i$的构造基于零点即可，



$$
L_i(x) = \frac{\prod _{j \ne i} (x - x_j)}{\prod_{j \ne i}(x_i -x_j)}
$$



基于上述这组基可以得到满足条件的插值多项式，



$$
p(x) = \sum_i y_i L_i(x)
$$



在计算中，为了减少重复计算量，可以如下定义$w(x),\lambda_i$并且提前计算，



$$
\begin{align}
&p(x) = w(x) \sum_i \frac{y_i}{\lambda_i(x-x_i)} \\
\text{With }& w(x) = \prod_j(x-x_j) \\
& \lambda_i = \prod_{j \ne i} (x_j - x_i)
\end{align}
$$


### Neville's Method

Neville插值方法的思想是用两个$n-1$个插值点构成的插值多项式，合成用$n$个插值点构成的插值多项式。

记$p_{1,2,...,n}(x)$为$x_1,x_2,...,x_n$些插值点构成的插值多项式，对于插值多项式$p_{2,...,n}(x)$和$p_{1,...,n-1}(x)$，想要用上述两个多项式的仿射组合得到插值多项式$p_{1,...,n}(x)$：



$$
p_{1,...,n}(x) = \alpha p_{2,...n}(x) + (1-\alpha) p_{1,...,n-1}(x)
$$



显然，对于$x_2,...x_{n-1}$，根据仿射组合的定义，其一定满足插值条件，只需考虑端点处的$x_1,x_n$，此时只要取


$$
\alpha = \frac{x-x_1}{x_n -x_1},1-\alpha = \frac{x_n -x}{ x_n -x_1}
$$



则可以保证组合后的多项式在所有点处都满足插值条件。

据此，我们得到Neville的插值方法，可以用行列式的形式表示为，



$$
p_{1,2,...,n}(x) = \frac{
det
\begin{pmatrix}
x -x_1 & p_{1,2,...,n-1}(x)\\
x - x_n & p_{2,3,...,n}(x)
\end{pmatrix}
}{x_n - x_1}
$$



在插值的过程中，通常考虑内插值，也即$ x_1 \le x \le x_n$, 此时有$0 \le \alpha \le 1$,也即相当于做了一个凸组合。

而对于凸组合的计算是数值稳定的，假设$d = \alpha  b+(1-\alpha) c$,而$b,c$的误差分别为$\delta(b),\delta(c)$,则计算出来凸组合$d$的数值误差满足：


$$
\delta(d) \le \alpha \delta(b) + (1-\alpha) \delta(c) \le \max(\delta(b),\delta(c))
$$


因此，进行凸组合过后的数值误差不会变大，此时Nevill插值方法是数值稳定的。



### Newton's Method



Newton插值巧妙地利用前$n-1$个插值点的插值多项式，递推得到$n$个结点的插值多项式。

此时只要，加上一个由前$n-1$个根组成的多项式$\prod_{j=1}^{n-1}(x -x_j)$,乘上一个组合系数$\alpha_{n-1}$



$$
p_{1,2,...,n} = p_{1,2,...n-1} + \alpha_{n-1} \prod_{j=1}^{n-1}(x -x_j) = \sum_i \alpha_i \prod_j^i (x-x_j)
$$



根据插值多项式的唯一性，且系数$\alpha_{n}$为该插值多项式的最高次系数，可以根据对照Lagrange多项式的最高次系数得到，



$$
\alpha_n = \sum_{k=1}^n \frac{y_n}{\prod_{j \ne k} (x_k -x_j)} 
$$



实际上，展开可以发现上述的$\alpha_n$其实还是$n$阶差分$f[x_1,x_2,...,x_n]$的表达式，$n$阶差分是Newton方法的精髓，其中$n$阶差分可以被$n-1$阶差分递归定义，且满足：



$$
\begin{align}
f[x_1] &= y_1 \\
f[x_1,x_2] &= \frac{y_2-y_1}{x_2 -x_1 }\\
f[x_1,x_2,x_3] &= \frac{\frac{y_3-y_2}{x_3 -x_2 }- \frac{y_2-y_1}{x_2 -x_1 }}{x_3 -x_1 } \\
&... \\
f[x_1,x_2,...,x_n] &= \frac{f[x_2,...,x_n]-f[x_1,...x_{n-1}]}{x_n-x_1}
\end{align}
$$



由于$n$阶差分非常重要，下面给出其性质，

$n$阶差分具有轮换对称性，对于任意排列$\sigma$,证明可以根据Lagrange插值多项式首项的轮换对称性得到：



$$
f[x_1,x_2,...,x_n] = f[\sigma(x_1),\sigma(x_2),...,\sigma(x_n)]
$$



$n$阶差分和微分也存在关系，当取极限的情况下:



$$
\lim_{(x_1,...,x_{n+1}) \rightarrow (x_0,...,x_0)} f[x_1,...x_{n+1}] = \frac{f^{(n)}(x_0)}{n !}
$$



## Polynomial Reconstruction

如果插值点来自于某个函数$f(x)$，此时多项式插值问题转换为多项式重构问题，也即我们希望寻找一个多项式尽可能重构出原函数$f(x)$.



### Error Bound

根据Newton插值方法，可以给出利用插值进行多项式重构的误差。

取$n$个插值点进行插值得到插值多项式$p_n(x)$之后，在其基础上去，对于任意函数值$f(x)$可以看作根据$n$个插值点构成的多项式，使用Newton方法新增一个插值结点形成的。


$$
f(x) - p_n(x) = f[x_1,x_2,...x_n,x] \prod_j^n (x-x_j)
$$


而由于差分和微分的关系，根据推广的中值定理，存在$\xi$满足：


$$
f(x) - p_n(x) = \frac{f^{(n)}(\xi)}{n!} \prod_j^n (x-x_j)
$$


因此，我们可以得到多项式重建问题的误差上界，


$$
\Vert f(x) - p_n(x) \Vert_{\infty} = \frac{\Vert  f^{(n)}(x) \Vert_{\infty}}{n!} \Vert \prod_j^n (x-x_j) \Vert_{\infty}
$$


### Runge Phenomenon

Runge现象说明了某些情况下，采取等距插值结点可能导致插值结果很差，甚至插值结点越多，插值误差越差的现象，或者


$$
\lim_{n \rightarrow \infty} \sup \Vert f(x) - p_n(x) \Vert_{\infty} > 0, \exists f(x)
$$




### Chebyshev Interpolation

为了避免Runge现象，在可以自由选择插值结点的情况下，通常采用Chebyshev结点插值，此时可以证明这种插值方法是最优的。


$$
\Vert f(x) - p_n(x) \Vert_{\infty} = \frac{\Vert  f^{(n)}(x) \Vert_{\infty}}{n!} \Vert \prod_j^n (x-x_j) \Vert_{\infty}
$$


根据重构误差上界，$\Vert  f^{(n)}(x) \Vert_{\infty}$和函数$f(x)$本身的性质相关，和插值结点相关的项为$\Vert \prod_j^n (x-x_j) \Vert_{\infty}$，考虑最小化该项，

首先考虑$x \in [-1,1]$的情况，考虑取插值结点满足，$x_j = \cos \frac{2j-1}{2n} \pi$，此时称为Chebychev结点，对比函数的根和首项系数并且根据插值多项式的唯一性可以知道，


$$
T_n(x) = 2^{n-1} \prod_j^n (x-x_j) = \cos (n \arccos x )
$$


上式定义的$T_n(x)$即为Chebyshev多项式，此时显然满足


$$
\Vert \prod_j^n (x-x_j) \Vert_{\infty} = \frac{\Vert T_n(x) \Vert_{\infty}}{2^{n-1}} =\frac{1}{2^{n-1}}
$$


下面我们证明由Chebshev多项式给出的上述$\Vert \prod_j^n (x-x_j) \Vert_{\infty} $的界是最小的界，

假设存在其他某些插值结点插值而成的$n$阶首一多项式$p(x) = \prod_j^n (x-x_j)$ 有更小的界,也即 

$$
\Vert \prod_j^n (x-x_j) \Vert_{\infty} < \frac{1}{2^{n-1}}
$$

构造函数$g(x) = p(x) - \frac{T_n(x)}{2^{n-1}}$ ,且已知Chebyshev多项式在$x_j = \cos \frac{j}{n} \pi$ 这些结点上取到极值，在这些极值点上，有


$$
g(x_j) = p(x_j) - \frac{(-1)^j}{2^{n-1}}
$$


根据假设，$\Vert p(x_j) \Vert_{\infty} < \frac{1}{2^{n-1}}$ ,因此$g(x)$在这$n+1$个结点上正负交错，因此其具有$n$个根。

又因为$g(x)$为两个首一$n$阶多项式之差，$g(x)$为$n-1$阶多项式，具有$n$个根只能为0多项式，矛盾。

因此，处理Chebyshev结点，不可能存在其他插值结点构成的插值多项式具有更小的界。

将Chebyshev多项式推广到任意闭区间$[a,b]$上，可以得到Chebyshev插值的误差上界，


$$
\Vert f(x) - p(x) \Vert_{\infty} \le \frac{ \Vert f^{(n)}(x) \Vert_{\infty}}{2^{n-1} n!}(\frac{b-a}{2})^n
$$




## Hermite Interpolation

上述问题中考虑的多项式插值都仅在函数值上提要求，Hermite插值对导数值提要求，也即要求插值得到的函数具有一定的光滑性。

### Transform to Newton‘s Method

Hermite插值实际上可以转化为一般的插值问题。

例如，基于Lagrange插值的思想，只需要找到一组高阶导数满足给定条件的基，就可以组成新的插值函数。

基于Newton插值方法，可以给出更简洁的Hermite插值的计算，也即将Hermite插值问题看作允许重结点的Newton插值即可。

例如，对于重结点$x_1,x_1$，相当于在$f[x_1,x_1] = \frac{f'(x_1)}{2}$上进行插值。



### Matrix Function

Hermite插值本质上和矩阵函数息息相关。

根据Hamilton-Cayley定理，任意矩阵$A$适合于其特征多项式$p(A)=O$

对于不熟悉Hamilton-Cayley定理的读者，以下通过归纳法简证，熟悉的读者可以跳过该部分：

---

归纳假设是对于$n-1$阶矩阵Hamilton-Cayley定理成立，

首先对矩阵$A$进行Schur分解，


$$
PAP^{-1} = 
\begin{pmatrix}
\lambda_1 & \star \\
0 & B \\
\end{pmatrix}
$$


根据归纳假设,$B$适合于其特征多项式$q(B)=O$,

因此，



$$
p(A) = (A- \lambda_1I) q(A) = P 
\begin{pmatrix}
0 & \star \\
0 & B
\end{pmatrix}

\begin{pmatrix}
q(\lambda_1) & \star \\
0 & O
\end{pmatrix}
P^{-1} = O\\
$$


Hamilton-Cayley定理得证。

---

根据Hamilton-Cayley定理，$A^n$可以被$I,A,...,A^{n-1}$线性表示

因此将任意矩阵函数$f(A)$进行Taylor展开后，其都可以被一个$n-1$阶多项式表示，也即$f(A) = p(A)$

根据Jordan标准型理论，求$p(A)$等价于求解$p(J)$，其中$J$为$A$的Jordan标准型。

而对矩阵$J$进行Taylor展开可以得到，


$$
f(J) = f(\lambda I+J-\lambda I) =f(\lambda I) + f'(\lambda I) (J -\lambda I) + \frac{f''(\lambda I)}{2} (J - \lambda I)^2 +... 
$$

代入可以得到，



$$
f(J) = 
\begin{pmatrix} 
\lambda  & f'(\lambda) & f''(\lambda) & ...\\
0 & \lambda & f'(\lambda) & f''(\lambda) \\
0 & 0  & \lambda  & f'(\lambda) \\
...& ...&...&...
\end{pmatrix}
$$


此时求解$f(J)$不仅要求函数值相等，也要求各阶导数值满足条件，也即相当于在$\lambda_1,\lambda_2,...\lambda_n$这些点上做Hermite插值。





## Spline Interpolation

样条插值也即使用分段函数进行插值，使用分段函数通常可以获得更小的插值误差上界。



### Linear Spline 

线性样条是一种简单的想法，在$x_0,x_1,...,x_n$个结点的端点处进行插值，也即使得$s(x_i) = f(x_i)$,且在中间使用线性函数分段拟合，得到样条函数$s(x)$,其插值误差上界，根据上面的理论，在每一段处$n=2$, 并且令区间长度为$h$,




$$
\Vert f(x) - s(x) \Vert_{\infty} = \frac{\Vert  f^{(n)}(x) \Vert_{\infty}}{n!} \Vert \prod_j^n (x-x_j) \Vert_{\infty}
$$



而对于



$$
\Vert \prod_j^n (x-x_j) \Vert_{\infty} = \Vert (x-x_1) (x- x_2) \Vert_{\infty} = \frac{h^2}{4}
$$



因此，可以得到线性样条插值的重构误差上界为，



$$
\Vert f(x) - s(x) \Vert_{\infty} = \frac{h^2}{8} \Vert  f^{(2)}(x) \Vert_{\infty}
$$




### Cubic Spline

一种更常用的方法是三次样条插值，其具有很好的性质。

定义多段在$[x_i,x_{i+1}]$上的分段三次函数$s_i(x) = a_i x^3 + b_i x^2 + c_i x +d_i$,使其函数值满足插值条件且一阶和二阶导数均连续:



$$
\begin{align}

s_i(x_i) &= f(x_i) \\
s_i(x_{i+1}) &= f(x_{i+1}) \\
s_i'(x_{i+1}) &= s_{i+1}'(x_{i+1})\\
s_i''(x_{i+1}) &= s_{i+1}''(x_{i+1})
\end{align}
$$



由于参数总的自由度为$4n$，但上述只提了$4n-2$个条件（端点处没有导数的联系性要求），通常对于端点处增加要求，根据要求的不同可以分为不同的样条：



* Complete，完全样条：指定了端点处的一阶导数值，$s_0'(x_0) = k_0, s_{n-1}'(x_{n})=k_n$
* Natural，自然样条：令端点处的二阶导数值为0，$s_0''(x_0) = 0, s_{n-1}''(x_{n})=0$
* Periodic，周期样条：令端点处的一阶和二阶导数满足周期性，$s_0'(x_0) = s_{n-1}'(x_{n}),s_0''(x_0) = s_{n-1}''(x_{n})$



构造相应的线性方程组，即可解得对应的参数取值，得到三次样条插值函数。



可以证明，三次样条通常可以最小化如下定义的势能，其中$t(x)$为满足插值条件的函数。


$$
E = \int_a^b [t''(x)]^2 dx
$$

对于任意$t(x)$, 定义$\eta(x) = t(x)  - s(x)$, 



$$
\begin{align}
\int_a^b [t''(x)]^2 dx &= \int_a^b [s''(x) + \eta''(x)]^2 dx\\
&=  \int_a^b [s''(x)]^2 dx + \int_a^b [\eta''(x)]^2 dx + \int_a^b s''(x) \eta''(x) dx 
\end{align}
$$



对于交叉项，对于每个分段使用分部积分公式，考虑到三次函数，$s'''(x)=0$,则有，



$$
\begin{align}
\int_a^b s''(x) \eta''(x) dx &=  \int_a^b s''(x) d \eta'(x)   \\
&= s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n} - \int_a^b s'''(x) \eta'(x) d x \\
&= s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n} - \int_a^b s'''(x)d\eta(x) \\
&= s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n} - s'''(x) \eta(x) \Big{\vert}_{x_0}^{x_n} +\int_a^b \eta(x)ds'''(x)  \\ 
&= s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n} - s'''(x)  \eta(x) \Big{\vert}_{x_0}^{x_n}
\end{align}
$$



由于样条的插值条件，$\eta(x_0) = \eta(x_n)=0$ ，因此：$ s'''(x)  \eta(x) \Big{\vert}_{x_0}^{x_n}=0$.

而对于自然样条，$s(x_0)'' = s''(x_n)=0$，因此： $s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n}=0$

而对于完全样条，如果限制函数$t(x)$也满足，$t'(x_0) = k_0, t'(x_n) = k_n$，则$\eta'(x_0) = \eta'(x_n) = 0$，

因此在限定的函数空间内，此时也有： $s''(x)  \eta'(x) \Big{\vert}_{x_0}^{x_n}=0$



综上，可以得到，




$$
\begin{align}
\int_a^b [t''(x)]^2 dx =  \int_a^b [s''(x)]^2 dx + \int_a^b [\eta''(x)]^2  \ge \int_a^b [s''(x)]^2 dx dx  
\end{align}
$$


也即，三次样条函数是所有$t(x)$的函数空间内最小化势能的函数。





