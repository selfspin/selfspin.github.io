---
title: '线性方程组的迭代解法'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 数值算法
​---


线性方程组的迭代解法。

<!--more-->



考虑使用数值方法求解线性方程组 $A  x= b$.



## Richardson Iteration

Richardson迭代法考虑使用不动点迭代求解上述方程，迭代格式形如



$$
\begin{align*}
x_{k+1} = x_k + (b - Ax_k).
\end{align*}
$$



这是一个不动点迭代，根据不动点迭代收敛的充要条件，我们知道上述算法收敛当且仅当



$$
\begin{align*}
\rho(I- A) <1.
\end{align*}
$$



考虑预处理的Richarson迭代，给定可逆矩阵 $M$, 我们可以等价地求解



$$
\begin{align*}
M^{-1} Ax = M^{-1} b .
\end{align*}
$$



对上述的预处理后的方程使用Richardson迭代算法得到



$$
\begin{align*}
x_{k+1} = x_k + M^{-1} ( b - Ax_k).
\end{align*}
$$



选取不同的预处理矩阵 $M$ 就可以得到不同的古典迭代算法，下面主要介绍三种算法并且分析其收敛性。

其中收敛性证明的部分参考自 [1].

将矩阵 $A$ 分解为对角部分以及上下三角部分， 也即 $A = D - L -U$.

如果选取 $M = D$, 我们得到了Jacobi迭代算法，简称J法，形如



$$
\begin{align*}
x_{k+1} = D^{-1} (b + L + U) x_k.
\end{align*}
$$



其分量形式为



$$
\begin{align*}
x_{k+1}^{(i)} = \frac{1}{a_{ii}} \left(b - \sum_{j=1}^{i-1} a_{ij} x_{k}^{(j)} - \sum_{j=i+1}^n a_{ij} x_k^{(j)} \right).
\end{align*}
$$



如果选取 $M = D- L$， 我们得到了Gauss-Seidel迭代，简称GS法，形如



$$
\begin{align*}
x_{k+1} = (D-L)^{-1} (b+U) x_k.
\end{align*}
$$



其分量形式为



$$
\begin{align*}
x_{k+1}^{(i)} =  \frac{1}{a_{ii}} \left(b - \sum_{j=1}^{i-1} a_{ij} x_{k+1}^{(j)} - \sum_{j=i+1}^n a_{ij} x_k^{(j)} \right).
\end{align*}
$$



其与J法进行对比，可以发现其区别在于按照分量顺序 $x_{k+1}^{(i)}$ 从 $i = 1,\cdots,n$ 逐次进行计算，但是对于第 $i$ 地分量的时候，所有前 $1,\cdots, i -1$ 个分量的对应值都代入新得到的值。GS法实际上是著名的SOR (Successive Over-Relaxation, 超松驰迭代算法）的一个特例，SOR法等价于选取 $M = (D-  \omega L) / \omega$. 这时候得到如下的迭代算法，



$$
\begin{align*}
x_{k+1} &= x_k + \omega ( D - \omega L)^{-1} ( b -A x_k) \\
&= x_k + \omega (D-\omega L)^{-1} \left( b - \left( \frac{D - \omega L}{\omega} - \frac{(1- \omega) D + \omega U}{\omega} \right) x_k \right) \\
&= \omega (D- \omega L)^{-1} b + (D- \omega L)^{-1} ((1- \omega) D + \omega U) x_k.
\end{align*}
$$



移项后得到



$$
\begin{align*}
(D - \omega L) x_{k+1} = \omega b + ((1- \omega ) D + \omega U) x_k.
\end{align*}
$$



写成分量形式如



$$
\begin{align*}
x_{k+1}^{(i) } =  (1- \omega) x_k^{(i)} + \frac{\omega}{a_{ii}} \left(b - \sum_{j=1}^{i-1} a_{ij} x_{k+1}^{(j)} - \sum_{j=i+1}^n a_{ij} x_k^{(j)} \right).
\end{align*}
$$



形式上等价于对于GS法以及 $x_k$ 进行了一个权重为 $\omega \in (0,1)$ 的加权和。当 $\omega =1$ 时即为GS法。



下面我们分析上述三种方法的收敛性。首先，我们可以直接观察到，若 $A$ 为严格对角占优矩阵，那么J法和GS法都一定收敛。我们可以证明两个方法对应的迭代矩阵都具有小于 $1$ 的谱半径，其中J法的迭代矩阵为 $M = D^{-1} (L+U)$, 而GS法的迭代矩阵为 $M = (D-L)^{-1}U$.

否则，假设其存在特征值 $\lambda$ 满足 $\vert \lambda \vert \ge 1$. 对于J法将得到 $ \det (D - \lambda^{-1} (L+U))=0$ ,而对于GS法将得到 $\det( D - L - \lambda^{-1}U) = 0$. 由于 $\vert \lambda \vert \ge 1$, 我们知道上面两个关于 $\lambda$ 的矩阵都为严格对角占优矩阵，其一定是非奇异的矩阵。矛盾。



上述分析说明了J法和GS法对于严格对角占优矩阵一定收敛，下面我们考虑 $A$ 为对称正定矩阵的情况。

此时我们知道$a_{ii}>0$, J法收敛当且仅当



$$
\begin{align*}
\rho(D^{-1} (D-A)) <1  \quad\Leftrightarrow\quad 0 < \lambda(D^{-1} A) <2 \quad\Leftrightarrow \quad 2D-A \succeq 0.
\end{align*}
$$



下面我们分析SOR方法，其结论取 $\omega = 1$ 即为 GS法的收敛性。对于SOR方法，其迭代矩阵为



$$
\begin{align*}
M = (D - \omega L)^{-1} \left((1-\omega) D + \omega U \right).
\end{align*}
$$



注意到



$$
\begin{align*}
\det(M) = \left( \det(D)\right)^{-1} \det ( (1-\omega)D) = (1- \omega)^n.
\end{align*}
$$

我们知道



$$
\begin{align*}
\rho(M) = \max_{1 \le i \le n} \vert\lambda_i(M) \vert  \ge \frac{1}{n} \sum_{i=1}^n \vert\lambda_i(M) \vert \ge \left \vert \prod_{i=1}^n \lambda_i(M) \right \vert^{1/n} = 1-\omega.
\end{align*}
$$



因此若SOR方法收敛，一定有 $\omega \in (0,2)$. 下面我们再证明这个条件是充分的。

设 $\lambda$ 为 $M$ 的一个特征值，我们有



$$
\begin{align*}
\lambda (D - \omega L) x  = ( (1- \omega) D + \omega U) x.
\end{align*}
$$



在上式两边同时对 $x$ 作内积得到



$$
\begin{align*}
\lambda x^\top D x - \lambda \omega x^\top L x = (1- \omega) x^\top D x + \omega x^\top Ux. 
\end{align*}
$$



设 



$$
\begin{align*}
x^\top D x = p, \quad x^\top L x =  \alpha + i \beta \quad \Rightarrow \quad x^\top U x = \alpha - i \beta.
\end{align*}
$$



代入后得到



$$
\begin{align*}
\lambda ( p -\omega ( \alpha + i \beta)) = (1- \omega) p + \omega (\alpha - i \beta).
\end{align*}
$$



两边同时取模后得到



$$
\begin{align*}
\vert \lambda \vert^2 = \frac{ (p - \omega( p -\alpha))^2 + (\omega \beta)^2 }{ ( p -\omega \alpha)^2 + (\omega \beta)^2}.
\end{align*}
$$



其模长小于 $1$ 等价于分子小于分母，将分母减去分子得到



$$
\begin{align*}
(\ast) = \omega p (p- 2\alpha ) (2 - \omega)
\end{align*}
$$



根据 $A$ 的正定性可知 $p -2 \alpha >0$, 因此当且仅当 $\omega \in (0,2)$ 时迭代矩阵的模长小于 $1$, SOR方法收敛。

作为 $\omega=1$ 的特例，上述结论说明了对于任意的正定矩阵$A$, GS法一定收敛。



## Chebyshev Iteration



回顾Richarson迭代，



$$
\begin{align*}
x_{k+1} &= x_k + (b - Ax_k) = (I-A) x_k + Ax^\ast \\
x_{k+1} - x^\ast &= ( I-A) (x_k - x^\ast) = (I-A)^m (x_0 - x^\ast).
\end{align*}
$$



注意到这为一个关于矩阵 $A$ 的多项式，我们自然地希望可以存在一个最优的多项式可以使得误差最小，有趣的试该问题正好等价于数值逼近中的经典问题。考虑非定常的迭代，其中迭代的步长为一个序列 $\{ \tau_k\}$.



$$
\begin{align*}
x_{k+1} = x_k +  \tau_{k+1} (b - Ax_k).
\end{align*}
$$



这将给出如下的误差递归式



$$
\begin{align*}
x_{k} - x^\ast = \prod_{j=1}^{k} (1 - \tau_j A) (x_0 - x^\ast) = p_k(A) (x_0 - x^\ast). 
\end{align*}
$$

那么



$$
\begin{align*}
\Vert x_k - x^\ast \Vert \le \Vert p_k(A) \Vert \Vert x_0 - x^\ast \Vert \le \sup_{\lambda \in [a,b]} \vert p_k(\lambda) \vert~ \Vert x_0 - x^\ast \Vert.
\end{align*}
$$



其中 $a,b$ 分别表示矩阵 $A$ 的特征值的下界和上界，$p_k(x)$ 为满足 $p_k(0) = 1$ 的 $k$ 次多项式，可以验证使得该逼近误差最小的多项式为Chebyshev多项式，也即



$$
\begin{align*}
p_k^\ast(t) = \arg \min_{p_k} \sup_{t \in[\lambda_n,\lambda_1] } \vert p_k(t) \vert = T_k\left( \frac{\lambda_1+\lambda_n - 2 t}{\lambda_1 -\lambda_n}  \right)  / T_k \left( \frac{\lambda_1+\lambda_n}{\lambda_1-\lambda_n} \right). 
\end{align*}
$$



假设存在另一个误差更小的多形式满足 $q_k(0)=1$. 考虑函数 $r = p-q$, 首先我们知道 $r(0) = 0$，并且 $p_k$ 在 $[a,b]$ 内存在 以一个大小为 $k+1$ 的极大交错组，由于 $q_k$ 的逼近误差更小，在这个极大交错点组处 $r$ 的符号全由 $p$ 的符号所决定，也即 $r$ 具有 $k+1$ 个极大交错点组，其之前必存在 $k$ 个零点，而 $r$ 为 $k$ 次多项式，只能存在至多 $k$ 个零点。矛盾。



通过Chebshev多项式的零点的表达式可以得到 $\tau_j$ 的取值满足



$$
\begin{align*}
\tau_j = \frac{b+a}{2}- \frac{b-a}{2} \cos \left( \frac{(2j-1) \pi}{2k} \right), \quad j = 1,\cdots,k.
\end{align*}
$$



上述导出的非定常迭代就称为Chebyshev迭代。根据Chebyshev多项式的性质，可以给出该方法的误差估计，根据上述分析，我们知道



$$
\begin{align*}
\Vert x_k - x^\ast \Vert \le T_k \left( \frac{\lambda_1+\lambda_n}{\lambda_1-\lambda_n} \right)^{-1} \Vert x_0 -x^\ast \Vert.
\end{align*}
$$



定义问题的条件数为 $\kappa = \lambda_1  / \lambda_n$. 根据 [4] 中的不等式，我们知道对于任意的 $\epsilon <1/2$ 成立



$$
\begin{align*}
T_k(1+ \epsilon) \ge \frac{(1+\sqrt{\epsilon})^k}{2}.
\end{align*}
$$
因此当 $\kappa >5$ 的时候我们有误差估计



$$
\begin{align*}
\Vert x_k - x^\ast \Vert \le 2 \left(1 - \frac{\sqrt{2}}{\sqrt{\kappa - 1} + \sqrt{2}} \right)^k \Vert x_0 - x^\ast \Vert, \quad \kappa >5.
\end{align*}
$$



为了达到 $\epsilon$ 最优解，其所需要的复杂度为 $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$.

上述给出的Chebyshev迭代格式需要预先选取步长序列 $\{\tau_k \}$, 另一种方法是使用Chebychev多项式的三项递归关系进行构造，



$$
\begin{align*}
T_{k+1}(z) = 2z T_k(z) - T_{k-1}(z).
\end{align*}
$$



考虑在 $z_0$ 处进行归一化，记归一化后的函数为 $\hat{T}_k(z)$.



$$
\begin{align*}
\hat T_{k+1}(z) = \frac{2z T_k(z_0)\hat T_k(z)}{T_{k+1}(z_{0})} - \frac{T_{k-1}(z_0) \hat T_{k-1}(z)}{T_{k+1}(z_0)}.
\end{align*}
$$



令 $z_0 = (\lambda_1+\lambda_n)/ (\lambda_1-\lambda_n)$. 考虑变换 $z = (\lambda_1+\lambda_n - 2t) / (\lambda_1-\lambda_n)$. 我们知道 $p_k^\ast(t) = \hat T_{k}(z)$. 那么



$$
\begin{align*}
p_{k+1}^\ast (t) &= \frac{2 z_0 T_k(z_0) }{T_{k+1}(z_0)} p_k^\ast(t) - \frac{T_{k-1}(z_0) }{T_{k+1}(z_0)} p_{k-1}^\ast(t) - \frac{4}{\lambda_1-\lambda_n} \frac{T_k(z_0)}{T_{k+1}(z_0)}  t p_k^\ast(t) \\
&= \left(1 +\frac{T_{k-1}(z_0)}{T_{k+1}(z_0)} \right) p_k^\ast(t) - \frac{T_{k-1}(z_0)}{T_{k+1}(z_0)} p_{k-1}^\ast(t) - \frac{4}{\lambda_1-\lambda_n} \frac{T_k(z_0)}{T_{k+1}(z_0)} t  p_k^\ast(t) \\
&= (1+\alpha_k) p_k^\ast(t) -\alpha_k p_{k-1}^\ast (t) - \beta_k  t p_k^\ast
(t)
\end{align*}
$$



可以递归地验证这等价于如下的迭代



$$
\begin{align*}
x_{k+1} &= p_{k+1}^\ast(A) (x_0 - x^\ast)  + x^\ast \\
&= (1+ \alpha_k) p_k^\ast(A)  (x_0 - x^\ast) - \alpha_k p_{k-1}^\ast(A) (x_0 - x^\ast) - \beta_k A p_k^\ast(A) (x_0- x^\ast) + x^\ast \\
&= (1+\alpha_k) x_k - \alpha_k x_{k-1} + \beta_k (b - Ax_k).
\end{align*}
$$



这就是Chebyshev迭代的二项递推形式。



## Steepest Descent



本节介绍最速下降方法，这也可以看作Richardson迭代的变步长版本，方法考虑优化如下的变分问题



$$
\begin{align*}
\min_x \varphi(x) = \frac{1}{2} x^\top A x  - b^\top x.
\end{align*}
$$



可以验证该方程的最小值就是线性方程组 $Ax = b$ 的解。令残差向量 $r = b - Ax$.

每次迭代中，指定一个搜索方向，方法生成序列 $x_{k+1} = x_k + \alpha_k p_k$， 其中 $\alpha_k$ 选取使得最小化 $\varphi(x_k+ \alpha p_k)$.

根据一阶最优性条件，可以求得



$$
\begin{align*}
\alpha_k = \frac{p_k^\top r_k}{p_k^\top A p_k}.
\end{align*}
$$



再考虑如下确定搜索方向 $p_k$, 由于$r_k$ 为负梯度方向，也即使得函数值下降最快速的方向，因此一种自然的选取方法是 $p_k = r_k$, 这就是所谓的最速下降方法。下面我们分析该算法的收敛率。证明参考自 [2]. 利用到如下引理



$$
\begin{align*}
\Vert P(A) x \Vert_A \le \max_{1 \le i \le n} \vert P(\lambda_i) \vert \Vert x \Vert_A,
\end{align*}
$$



其中 $\lambda_1, \cdots ,\lambda_n$ 是矩阵 $A$ 的特征值。

根据最速下降方法的定义，我们知道对于任意的 $\alpha \in \mathbb{R}$, 成立



$$
\begin{align*}
&\quad \Vert x_{k+1} - x^\ast \Vert_A \\&= \sqrt{(x_{k+1} - x^\ast)^\top A (x_{k+1} - x^\ast)} \\
&\le \sqrt{(x_k + \alpha r_k - x^\ast)^\top A (x_k + \alpha r_k - x^\ast)} \\
&= \Vert (I-\alpha A) (x_k - x^\ast) \Vert_A   \\
&\le \min_\alpha \sup_{t \in [\lambda_n, \lambda_1]} \vert 1- \alpha t \vert ~\Vert x_k - x^\ast \Vert_A. 
\end{align*}
$$



注意到上式的左端等价于一个最优的一次多项式逼近问题，容易知道其解在 $\alpha = 2 /(\lambda_1 + \lambda_n)$ 处取到

代入后得到误差估计为



$$
\begin{align*}
\Vert x_{k} - x^\ast \Vert_A \le \left( 1 - \frac{2}{\kappa+1}\right) \Vert x_{k-1} -x^\ast \Vert_A \le \left( 1 - \frac{2}{k+1}\right)^k \Vert x_{0} - x^\ast \Vert_A.
\end{align*}
$$



为了达到 $\epsilon$ 最优解，其所需要的复杂度为 $\mathcal{O}(\kappa \log(1/\epsilon))$.

将其与Chebyshev迭代相比，发现其复杂度更高，下面我们将通过改进最速下降算法中的搜索方向 $p_k$, 得到著名的共轭梯度法，其将具有和Chebyshev迭代相同的收敛速率。



## Conjugate Gradients



我们首先介绍共轭梯度发的思想，并且启发式引入该算法，后面再给出严谨地收敛性证明。

我们在最速下降算法中，每一步选取最优的 $\alpha_k$ 使得 $x_{k+1}$ 是 $x_k$ 沿着方向 $p_k$ 极小化搜索的结果，但我们希望这同时也是在子空间 ${\rm span}(p_0,\cdots,p_k) $中极小化搜索的结果。我们将说明，这个雄心壮志是可以被精心选择的 $p_k$ 实现的。考虑



$$
\begin{align*}
x_{k+1} = x_0 + y + \alpha_k p_k , \quad y \in {\rm span}(p_0,\cdots p_{k-1}).
\end{align*}
$$



根据



$$
\begin{align*}
\varphi(x_{k+1}) & = \varphi(x_0 + y + \alpha_k p_k) \\
&= \varphi(x_0 +y )+ \alpha_k  y^\top A p_k  - \alpha_k b^\top p_k + \frac{\alpha_k^2}{2} p_k^\top A p_k. 
\end{align*}
$$



我们希望 $y^\top A p_k = 0$， 那么此时给定 $p_k$ 之后选取最优的 $y$ 和最优的 $\alpha_k$ 为两个可分的问题。进一步，如果我们知道 $x_0 +y$ 已经在子空间 ${\rm span}(p_0,\cdots, p_{k-1})$ 中极小化 $\varphi(x_0+y)$, 我们只需要如之前一样贪心地选取 $\alpha_k$ 即可。因此，仅需要搜索方向 $p_k$ 满足



$$
\begin{align*}
p_i^\top A  p_j = 0, \quad \forall i \ne j. 
\end{align*}
$$



满足上述条件的方向称为共轭方向。共轭梯度方法的核心在于用简单的迭代序列生成上述共轭方向。

下面我们将证明, 只需要简单地选取 $p_{k+1} = r_{k+1} + \beta_{k} p_{k}$ 就可以满足上述要求。利用 $p_{k+1}$ 与 $p_k$ 的共轭性质，我们知道



$$
\begin{align*}
\beta_k = -\frac{r_{k+1}^\top A p_k}{p_k^\top A p_k}.
\end{align*}
$$



下面我们通过归纳法证明如下两个事实


$$
\begin{align*}
r_i^\top r_j = 0, \quad p_i ^\top A p_j = 0, \quad \forall i \ne j.
\end{align*}
$$


也即剩余向量组构成一个正交向量组，而搜索方向构成一个共轭向量组。

首先我们验证归纳假设，初始化 $r_0 = p_0$. 那么


$$
\begin{align*}
r_0^\top r_1 &= r_0^\top r_0 - \alpha_0 r_0^\top A r_0 = p_0^\top r_0 - \alpha_0 p_0^\top A p_0 = 0 \\
p_1^\top A p_0 &= r_1^\top A p_0 + \beta_0 p_0^\top A p_0 = 0.
\end{align*}
$$


下面假设所要证明得结论对于 $r_1,\cdots, r_k$ 以及 $p_0,\cdots p_k$ 都成立，

根据迭代公式我们知道 $r_{k+1} = r_k - \alpha_k Ap_k$, 那么我们有


$$
\begin{align*}
r_{k+1}^\top p_k &= r_k^\top p_k - \alpha_k p_k^\top A p_k = 0 \\
r_k^\top p_k &= r_k^\top r_{k} + \beta_{k-1} r_k^\top p_{k-1}  = r_k^\top r_k.
\end{align*}
$$


利用上述结论，我们进而有


$$
\begin{align*}
r_{k+1}^\top r_j  = r_k^\top r_j - \alpha_k p_k^\top A r_j .
\end{align*}
$$


若 $j = k$ , 我们知道


$$
\begin{align*}
r_{k+1}^\top r_k &= r_k^\top r_k - \alpha_k p_{k}^\top A r_k \\
&= r_k^\top r_k - \alpha_k p_k^\top A(p_{k} - \beta_{k-1} p_{k-1} ) \\
&= r_k^\top p_k - \alpha_k p_k^\top A p_k \\
&= 0.
\end{align*}
$$


若$ j=0,\cdots, k-1$, 由归纳假设，


$$
\begin{align*}
r_{k+1}^\top r_j &=  r_k^\top r_j - \alpha_k p_k^\top A r_j  \\
&= - \alpha_k p_k^\top A (p_j - \beta_{j-1} p_{j-1}) \\
&= 0.
\end{align*}
$$


这就完成了 $r_0, \cdots, r_{k+1}$ 正交性的归纳，下面分析 $p_1,\cdots,p_{k+1}$ 的共轭性。

若 $k = j$, 根据 $\beta_k$ 的定义我们知道 $p_{k+1}^\top A p_k$.   若 $j = 0,\cdots,k-1$, 有


$$
\begin{align*}
p_{k+1}^\top A p_j &= r_{k+1}^\top A p_j + \beta_k p_k^\top A p_j \\
&= r_{k+1}^\top A p_j \\
&= \frac{1}{\alpha_k }r_{k+1}^\top (r_{j} - r_{j+1}) \\
&= 0.
\end{align*}
$$


最后一步利用到了已经证明的 $r_0,\cdots,r_{k+1}$ 的正交性，这就完成了归纳。

上述性质也可以用来化简 $\beta_k$ 的表达式:


$$
\begin{align*}
\beta_k &= - \frac{r_{k+1}^\top A p_k}{p_k^\top A p_k} = - \frac{r_{k+1}^\top A (r_k - r_{k+1})}{\alpha_k p_k^\top A p_k} = \frac{r_{k+1}^\top r_{k+1}}{ r_k^\top r_k}.
\end{align*}
$$


总结，我们得到了如下的共轭梯度法 (Conjugate Gradient, 简称CG法)的迭代公式如


$$
\begin{align*}
\alpha_k &= \frac{r_k^\top r_k}{p_k^\top A p_k}, \\
x_{k+1} &= x_k + \alpha_k p_k, \\
r_{k+1} &= r_k - \alpha_k A p_k , \\
\beta_k &= \frac{r_{k+1}^\top r_{k+1}}{ r_k^\top r_k}, \\
p_{k+1} &= r_{k+1} + \beta_k p_k.
\end{align*}
$$


最后，我们给出CG法的收敛性证明。该部分证明参考自[2].

可以根据归纳法证明向量组 $\{ r_k\}$ 和 $\{ p_k\}$ 所张成的实际上同为如下的Krylov子空间


$$
\begin{align*}
{\rm span}(r_0,\cdots, r_k) =  {\rm span}(p_0,\cdots, p_k) =  {\rm span}(r_0, Ar_0, \cdots, A^k r_0)=\mathcal{K}^{k+1}.
\end{align*}
$$


首先归纳假设显然成立，假设对于 $j = 1,\cdots k$ 都成立， 对于 $j = k+1$，我们知道


$$
\begin{align*}
r_{k+1} &= r_k - \alpha_k A p_k \in \mathcal{K}^{k+2} \\
p_{k+1} &= r_{k+1} + \beta_k p_k \in \mathcal{K}^{k+2}.
\end{align*}
$$


因此CG法实际上在子空间内极小化 $\varphi(x)$. 根据迭代公式，我们进一步知道 $x_{k} \in x_0 +\mathcal{K}^{k}$. 那么存在系数 $\mu_0,\cdots \mu_{k-1}$ 使得


$$
\begin{align*}
x_{k} &= x_0 + \mu_0 r_0 + \mu_1 A r_1 +  \cdots + \mu_{k-1} A^{k-1} r_0 \\
x_k - x^\ast &= x_0 - x^\ast + \mu_0 r_0 + \mu_1 A r_1 +  \cdots + \mu_{k-1} A^{k-1} r_0  \\
&= A^{-1}( r_0 + \mu_0 A  r_0 + \mu_1 A^2 r_0 + \cdots + \mu_{k-1} A^k r_0 ) \\
&= A^{-1} p_k(A) r_0.
\end{align*}
$$


其中 $p_k(t)$ 是一个满足 $p_k(0) = 1$ 的 $k$ 次多项式，由于CG法在Krylov子空间中最小化 $\varphi(x)$, 也即最小化 $\Vert x_k -x^\ast \Vert_A  $. 则


$$
\begin{align*}
\Vert x_k - x^\ast \Vert_A &= \min \left \{ \Vert x - x^\ast \Vert_A : x \in x_0 + \mathcal{K}^{k}  \right\} \\
&= \min_{p_k} \Vert A^{-1} p_k(A) r_0  \Vert_A \\
&\le \min_{p_k} \sup_{t \in [\lambda_n, \lambda_1]} \vert p_k(t) \vert ~  \Vert A^{-1} r_0 \Vert_A \\
&\le T_k \left( \frac{\lambda_1+\lambda_n}{\lambda_1-\lambda_n} \right)^{-1} \Vert x_0 - x^\ast \Vert_A.
\end{align*}
$$


这就得到了和Chebyshev迭代相似的结果，唯一的区别是范数采用 $A$-范数而非 $2$-范数，最终将得到如下的收敛率




$$
\begin{align*}
\Vert x_k - x^\ast \Vert_A \le 2 \left(1 - \frac{\sqrt{2}}{\sqrt{\kappa - 1} + \sqrt{2}} \right)^k \Vert x_0 - x^\ast \Vert_A, \quad \kappa >5.
\end{align*}
$$


为了达到 $\epsilon$ 最优解，其所需要的复杂度为 $\mathcal{O}(\sqrt{\kappa} \log(1/\epsilon))$，优于最速下降方法的 $\mathcal{O}(\kappa log(1/\epsilon))$ 的界。





## Reference



[1] 关冶、陆金甫，数值分析基础（第三版）

[2] 徐树方、高立、张平文，数值线性代数（第二版）

[3] 蒋尔雄、赵风光、苏仰锋，数值逼近（第二版）
