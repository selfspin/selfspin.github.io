---
title: '矩阵乘法的复杂度下界'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - 矩阵乘法
  - 理论计算机科学
  - 复杂度下界
---

Paper Reading: On the complexity of matrix product (STOC'02).

<!--more-->

给定两个 $m \times m$ 大小的矩阵，我们希望计算其矩阵乘法，得到一个大小也为 $m \times m$ 的矩阵，本文给出了第一个超线性的下界，为 $\Omega(m^2 \log m)$. 

首先定义我们的计算电路模型：电路 $C$ 是一个有向无环图，入度为0的结点表示输入结点，出度为0的结点表示输出结点，中间结点的入度为2，表示一个二元的乘法或者加法操作, 电路中每条边有一个标量表示一个传递值的时候的scale系数。电路的大小 （图的总边数）称为${\rm Size}(C)$ 表示整个算法的整体复杂度，电路的深度 (图中输入结点到输出结点的最长路) ${\rm Depth}(C)$  表示整个算法在并行意义下的最大运行时间。

如果电路中每条边对应的系数绝对值都小于1，我们称该电路为有界电路。如果一个电路只由加法门构成，我们称该电路为线性电路。对于矩阵乘法，输入为矩阵X和矩阵Y，对应的输出关于X和Y都为线性的，我们称这样的算子为双线性算子。为了给出关于双线性算子的一个下界，我们需要如下关于双线性电路的定义：该电路由三层电路组成，第一层和第三层为线性电路，只由加法门构成，中间一层计算双线性形式，由乘法门构成。第一层 （后面也称为输入层）关于$X$ 和 $Y$ 分别计算一个线性映射，然后中间层计算 $X$ 和 $Y$ 对应的部分的乘法，最后一层（后面也称为输出层）对运算结果再做一个线性映射。

注意到任何计算双线性映射的算法都可以表示为如上的双线性电路，这是因为由于双线性性质，每个结点做了乘法之后就不可能再接着进行乘法，因此乘法操作只会出现在中间的操作层。

## 有界线性电路的复杂度下界

我们首先研究关于有界线性电路的下界。给定关于自变量 $z_1,\cdots,z_n$ 的 $k$ 的线性函数 $L_1,\cdots,L_k$. 每一个 $L_i$ 可以看作 $\mathbb{R}^d$ 空间中的一个向量，我们定义 ${\rm Vol}_r [L_1,\cdots,L_k]$ 为 $\{L_1,\cdots,L_k \}$ 中的 $r$ 个向量以及任意其他 $n-r$ 个单位向量所张成的最大体积。也即，

$$
\begin{align*}
{\rm Vol}_r[L_1,\cdots,L_k] = \max_{i_1,\cdots,i_r, e_{r+1}, \dots e_{n}} \vert \det [ L_{i_1},\cdots, L_{i_r}, e_1,\cdots, e_n] \vert.
\end{align*}
$$

类似地，对于$n \times n$ 的矩阵 $H$, 我们定义 ${\rm Vol}_r[H]$ 为其所有行 $L_1,\cdots,L_n$ 的r维体积：

$$
\begin{align*}
{\rm Vol}_r[H] = {\rm Vol}_r[L_1,\cdots,L_k].
\end{align*}
$$

对于计算线性映射 $L_1,\cdots,L_k$ 的有界线性电路 $C$, 我们在本节中证明如下的不等式

$$
{\rm Size}(C) \ge \log_2 ({\rm Vol}_r[L_1,\cdots,L_k]), \quad \forall 1 \le r \le k.
$$

定义 $s = {\rm Size}(C)$. 由于电路 $C$ 为一个有向无环图，该电路的计算顺序实际上定义了一族线性函数 $f_1,\cdots,f_n, f_{n+1},\cdots, f_{n+s}$, 其中 $f_i= z_i$ ($1 \le i \le n$) 表示电路的输入。对于任意的 $i >0$ 我们知道存在 $i_1,i_2<i$ 以及绝对值小于1的系数 $c_1,c_2$ 使得 $f_i= c_1 f_{i_1} + c_2 f_{i_2}$.  根据行列式 （关于某一行）的线性性质，我们知道

$$
\begin{align*}
{\rm Vol}_r[f_1,\cdots,f_s,f_{s+1},\cdots,f_{s+i}] \le 2 {\rm Vol}_r[f_1,\cdots,f_s,f_{s+1},\cdots,f_{s+i-1}].
\end{align*}
$$

由于 ${\rm Vol}_r[f_1,\cdots,f_n] = 1$ 我们知道 

$$
\begin{align*}
{\rm Vol}_r[L_1,\cdots,L_k] \le {\rm Vol}_r[f_1,\cdots,f_s,f_{s+1},\cdots,f_{s+n}] \le 2^s.
\end{align*}
$$

我们定义如下的“刚性”：

$$
\begin{align*}
{\rm Rig}_r[L_1,\cdots,L_k] = \min_V \max_i {\rm dist}[L_i,V], \quad \forall $1 \le r \le n,
\end{align*}
$$

其中 $V$ 为 $r$ 维向量空间。类似地, 对于由行 $L_1,\cdots,L_n$ 所构成的矩阵 $H$, 我们可以定义 ${\rm Rig}_r[H] = {\rm Rig}_r[L_1,\cdots,L_n]$. 我们用下面的关系式连接我们所定义的体积和刚性：

$$
\begin{align*}
\log_2 ({\rm Vol}_r[L_1,\cdots,L_k]) \ge r \log_2({\rm Rig}_r[L_1,\cdots,L_k]).
\end{align*}
$$

不失一般性，我们假设 $r<k$ (否则显然有 ${\rm Rig}_r[L_1,\cdots,L_k] = 0$). 我们用如下的方式对 $L_1,\cdots,L_k$ 进行重排序：令 $L_1$ 为 $\{ L_1,\cdots,L_k\}$ 中最大化 ${\rm Vol}_1[L_i]$ 的向量 $L_i$, $L_2$ 为 $\{ L_2,\cdots,L_k\}$ 中最大化 ${\rm Vol}_1[L_1，L_i]$ 的向量 $L_i$， 以此反复进行下去... 

定义每次的体积的相对变化量 $v_i = {\rm Vol}_r[L_1,\cdots,L_i] / {\rm Vol}_{i-1}[L_1,\cdots,L_{i-1}]$. 根据我们的构造，实际上我们每次都在寻找距离原来所张成的空间最远的向量，那么根据体积公式，有

$$
\begin{align*}
v_{r+1} = \max_i {\rm dist}(L_i,V),\quad {\rm where}~~ V = {\rm Span}[L_1,\cdots,L_r].
\end{align*}
$$

从上式右端容易看出，$\{v_i \}_{i=1}^k$ 构成了一个单调递减数列。因此


$$
\begin{align*}
{\rm Vol}_r[L_1,\cdots,L_k] \ge {\rm Vol}_r[L_1,\cdots,L_r] = \prod_{i=1}^r v_i \ge (v_{r+1})^r \ge ({\rm Rig}_r[L_1,\cdots,L_k])^r.
\end{align*}
$$


上面所构造的工具可以用来给出关于线性映射的复杂度下界。我们希望将类似的技术推广到矩阵乘法上面，给定两个 $m \times m$ 的矩阵 $X,Y$, 对于给定的矩阵 $Y$, 我们实际上在计算一个关于 $X$ 的线性映射，该线性映射对应的矩阵为 $I \otimes Y$, 维度为 $m^2 \times m^2$. 根据行列式的性质 $\det[I \otimes Y] = (\det[Y])^m$. 根据我们提及的定义，我们知道，

$$
\begin{align*}
{\rm Vol}_{r \cdot m} [I \otimes Y] \ge ({\rm Vol}_r[Y])^m.
\end{align*}
$$


因此对于表达函数为 $I \otimes Y$ 的线性映射的有界线性电路，成立

$$
{\rm Size}(C) \ge r \cdot m \cdot \log_2({\rm Rig}_r[Y]).
$$

这是因为

$$
{\rm Size}(C) \ge \log_2({\rm Vol}_{r \cdot m}[I \otimes Y]) \ge m \log_2({\rm Vol}_r[Y]) \ge r \cdot m \cdot \log_2({\rm Rig}_r[Y]).
$$

## 矩阵乘法的复杂度下界

我们证明使得存在一个 $m \times m$ 的矩阵 $Y$ 使得一个计算 $X,Y$ 的矩阵乘法的双线性有界电路 $C$, 其大小不可能小于 $0.001 \cdot m^2 \log_2 m$. 否则， 我们知道对于双线性电路中输入层关于 $Y$ 的线性变换满足

$$
\begin{align*}
r \cdot m \cdot \log_2({\rm Rig}_{r \cdot m}[L_1,\cdots,L_k]) < 0.001 m^2 \log_2 m.
\end{align*}
$$

令 $r = m/10$, 得到

$$
\begin{align*}
{\rm Rig}_{r \cdot m} [L_1,\cdots, L_k] < m^{1/100}.
\end{align*}
$$

我们将在下一节给出引理，构造出矩阵 $Y$，满足如下的性质：

1. 考虑关于自变量 $Y_{1,1},\cdots,Y_{m,m}$ 的 $k$ 个线性函数 $L_1,\cdots,L_k$, 对于足够大的 $m$ (也即只需要对于常数 $m_0$ 满足 $m > m_0$)， 以及 $r=m/10$, 成立

$$
\begin{align*}
\vert L_i(Y_{1,1},\cdots,Y_{m,m}) \vert \le {\rm Rig}_{r \cdot m} [L_1,\cdots,L_k] \cdot (2 \ln k + 10)^{1/2}, \quad 1 \le i \le k.
\end{align*}
$$

2. ${\rm Rig}_r[Y] \ge \sqrt{m/9}$.

利用上面的引理，我们知道电路的大小为 $\mathcal{O}(m^2 \log_2(m))$, 那么 $k$ 的大小也不能超过该电路的大小，因此对于足够大的 $m$, 成立


$$
\begin{align*}
\vert L_i(Y_{1,1},\cdots,Y_{m,m}) \vert \le m^{1/100} \cdot (2 \ln k + 10)^{1/2}  < m^{1/99}.
\end{align*}
$$

对于电路 $C$, 如果固定矩阵 $Y$, 我们得到了关于矩阵 $X$ 的线性电路，但是矩阵 $Y$ 的元素并不都小于1，该线性电路并非有界线性电路。下面我们将其转换为一个有界线性电路, 注意到 $Y$ 的输入层经过了一次线性变换，每个元素绝对值小于 $m^{1/99}$, 我们将这些元素都除以 $m^{1/99}$ 这样与 $X$ 作用的时候都有有界线性算子。为了保持电路的输出值不变，我们需要在电路输出的时候，将输出的所有的 $m^2$ 个元素都乘上系数 $m^{1/99}$。 这可以通过 $\log_2(m^{1/99})$ 个有界的加法门来实现，这样子得到一个有界线性电路和原电路的输出值相同，其大小不超过 $(1/98) \cdot m^2 \log m$.

然而根据上一节中关于有界线性电路的复杂度下界，我们知道对于表达函数为 $I \otimes Y$ 的线性映射的有界线性电路，其大小必须大于等于

$$
\begin{align*}
r \cdot m \log_2( {\rm Rig}_r[Y]) \ge (1/20) \cdot m^2 \log_2(m/9).
\end{align*}
$$

这产生了矛盾。因此证明实现输出一个矩阵乘法的有界双线性电路，其大小至少为 $\Omega(m^2 \log m)$.

## 矩阵 $Y$ 的构造 (主引理的证明)

本节证明存在一个矩阵 $Y$ 满足引理所要求的条件。定义 $R = {\rm Rig}_{r \cdot m}[L_1,\cdot,L_k]$. 根据“刚性”的定义，我们知道存在 $r \cdot m$ 维的空间 $V$ 使得对于所有的 $1 \le i \le k$ 都有 ${\rm dist}(L_i,V) \le R$. 我们将没以恶搞向量 $L_i$ 都按照 $V$ 以及其正交补空间做分解， 得到 $L_i = L_i'' + L_i'$, 其中 $L_i'' \in V$, $L_i' \in V^{\perp}$. 由于 $\Vert L_i' \Vert = {\rm dist}(L_i,V)$ 我们知道对于 所有的 $1 \le i \le k$ 都有 $\Vert L_i' \Vert \le R$.

下面我们构造矩阵 $Y$, 我们让矩阵 $W$ 的每个元素都满足独立同分布的标准正态分布，然后定义矩阵 $Y$ 为 $W$ 在空间 $V^{\perp}$ 上面的投影。那么我们知道每一个 $L_i(Y)$ 都服从均值为0，方差为 $\Vert L_i' \Vert$ 的高斯分布。因此根据集中不等式的结果，我们知道对于所有的 $k$, 高概率 (以至少0.98的概率）成立 $\vert L_i(Y) \vert \le R (2 \ln k + 10)^{1/2}$. 这说明按该方法采样出的矩阵高概率满足主引理中的第一条性质。

下面我们证明第二条性质。我们首先证明，对于足够大的 $m$, 对于任意的 $m \times m$ 并且秩为 $r$ 的矩阵，以高概率 (以至少0.97的概率)成立 $\Vert Y - D \Vert \ge m/3$. 注意到我们使用的是矩阵的F-范数。为了证明该论断，我们考虑

$$
\begin{align*}
\Vert W -D \Vert^2 = \sum_{i=1}^m (\sigma_i(W) - \sigma_i(D))^2 \ge \sum_{i=r+1}^m (\sigma_i(W))^2 = \Vert W \Vert^2 - \sum_{i=1}^r (\sigma_i(W))^2.
\end{align*}
$$

根据经典的集中不等式的结论，我们知道 $\Vert W \Vert^2 = \Omega(m^2)$ 以及 $\sigma_1(W) = \mathcal{O}(\sqrt{m})$ 都以高概率成立，因此以高概率成立

$$
\begin{align*}
\Vert W -D \Vert^2 \ge \Vert W \Vert^2 - \sum_{i=1}^r (\sigma_i(W))^2 \ge \Vert W \Vert^2 - \sum_{i=1}^r (\sigma_1(W))^2= \Omega(m^2).
\end{align*}
$$

下面我们考虑 $\Vert W-Y \Vert^2$, 根据定义其为 $r m$ 个标准正态分布的平方和，同样根据经典的集中不等式的结果，以高概率成立 $\Vert W - Y \Vert = \mathcal{O}(\sqrt{r m})$.  最后根据距离的三角不等式，我们知道

$$
\begin{align*}
\Vert Y -D \Vert \ge \Vert W - D \Vert - \Vert W - Y \Vert > m/3,
\end{align*}
$$

其中 $1/3$ 仅仅代表某个常数因子。

那么如果 ${\rm Rig}_r[Y] < \sqrt{m / 9}$, 那么定义矩阵 $D$ 为 $Y$ 的所有行投影到空间 $V$ 后的行构成的矩阵，我们知道 $Y-D$ 的所有行的2-范数都小于 $\sqrt{m/9}$, 因此整个矩阵的F-范数满足 $\Vert Y- D \Vert^2 < m \cdot m/9 =(m/3)^2$, 这就导致了矛盾。

因此，主引理的第二条性质也应该以高概率满足。也就是说，根据我们所描述的采样方式，构造出的随机矩阵 $Y$ 以高概率满足引理的两条性质。这就说明了满足引理性质的矩阵 $Y$ 是一定存在的 （该事件发生的概率不可能为0），这就完成了证明。

