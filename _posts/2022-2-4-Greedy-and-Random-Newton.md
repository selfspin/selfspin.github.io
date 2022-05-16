---
title: 'Greedy and Random Quasi-Newton Methods'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 牛顿法
---





论文阅读笔记: [Greedy and Random Quasi-Newton Methods with Faster Explicit Superlinear Convergence](https://proceedings.neurips.cc/paper/2021/hash/347665597cbfaef834886adbb848011f-Abstract.html)



<!--more-->



## Quasi-Newton Update



考虑经典的伪牛顿法的更新，其目的是利用迭代序列产生的矩阵 $B_k$ 近似矩阵 $A$, 在优化算法中为 $A$ 可以为Hessian矩阵，


$$
\begin{align}
\text{SR1: } G_{k+1} &= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u}, \forall u \\
\text{BFGS: } G_{k+1} &= G_k - \frac{G_k u u^\top G_k}{u^\top G_k u} + \frac{A uu^\top A}{u^\top A u } ,\forall u  \\
\text{DFP: } G_{k+1} &= G_k -\frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u } + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top A u} ,\forall u
\end{align}
$$

### BFGS



**Greedy Version**



考虑衡量矩阵近似水平的指标，


$$
\begin{align}
\sigma_A(G) = tr(A^{-1}(G-A))  = tr(A^{-1}G ) -n
\end{align}
$$


根据BFGS的更新公式知道，


$$
\begin{align}
\sigma_A(G_{k+1}) &= \sigma_A(G_k) - \frac{u^\top G_k A^{-1} G_k u}{u^\top G_k u } + 1
\end{align}
$$


贪心版本算法从一组标准正交基 $u_j$ 中每次选取使得 $\sigma_A$ 下降最快的方向进行更新，也即，


$$
\begin{align}
\max_{u} \frac{u^\top G_k A^{-1} G_k u}{u^\top G_k u } = \max_{v} \frac{v^\top L A^{-1} L^\top v}{v^\top G_k v }, \text{Let } G_k = L^\top L, v = L u
\end{align}
$$


对于进行换元并且考虑 $ v = Lu$ 从标准正交基 $e_i$ 中选取，可以证明算法关于 $\sigma_A$ 线性收敛，


$$
\begin{align}
\sigma_A(G_{k+1}) &= \sigma_A(G_k) - \max_{e_i} \frac{e_i^\top L A^{-1} L^\top e_i}{e_i^\top  e_i } + 1 \\
&\le \sigma_A(G_k) - \frac{1}{n} tr(L A^{-1} L^\top) + 1 \\
&= \sigma_A(G_k) - \frac{1}{n} tr(A^{-1} G_k) +1 \\
&=(1- \frac{1}{n}) \sigma_A(G_k)
\end{align}
$$



递推则得到了，


$$
\begin{align}
\sigma_A(G_k)) \le (1- \frac{1}{n})^k \sigma_A(G_0)
\end{align}
$$

**Random Version**



考虑随机版本的算法，从标准证他分布中随机抽样随机向量  $\mathbf{u} = [u_i]$ , 根据正态分布的概率密度，容易证明其指向任意一个方向的概率相等，因此将其归一化之后服从 $n$ 维单位超球面上的均匀分布，此时有 ,


$$
\begin{align}
\mathbb{E} [ \frac{u u^\top}{u^\top u}] = \frac{1}{n } I_n
\end{align}
$$


因此，


$$
\begin{align}
\mathbb{E} [\sigma_A(G_{k+1})] &= \sigma_A(G_k) - \mathbb{E}_{u} [\frac{u^\top L A^{-1} L^\top u}{u^\top  u }] + 1 \\
&= \sigma_A(G_k) - tr(L A^{-1}L^\top) \mathbb{E} [ \frac{uu^\top}{u^\top u}] +1 \\
&= \sigma_A(G_k) - \frac{1}{n} tr(L A^{-1}L^\top)  +1 \\
&= (1- \frac{1}{n}) \sigma_A(G_k)
\end{align}
$$





### SR1

**Greedy Version**

针对SR1，需要考虑另外一种衡量矩阵近似的指标，定义为，


$$
\tau_A (G) = tr(G-A)
$$


且在优化函数满足 $L$- 光滑和 $\mu$ -强凸的假设下，上述两种指标反映的内容基本一致，


$$
\begin{align}
\mu I_n &\preceq A \preceq L I_n \\
\frac{1}{L} I_n &\preceq A^{-1} \preceq \frac{1}{\mu} I_n \\
\frac{1}{L} \tau_A(G) &\preceq \sigma_A(G) \preceq \frac{1}{\mu} \tau_A(G)
\end{align}
$$


根据SR1的更新公式，知道其性质，


$$
G_{k+1} u _k = A u_k
$$


根据更新公式，


$$
\begin{align}
\tau_A(G_{k+1}) &= \tau_A(G_k) - tr(\frac{u^\top (G_k-A)^2 u}{u^\top (G_k - A) u} ) =  \tau_A(G_k) - tr(\frac{v^\top (G_k-A) v}{v^\top v} ) ,\text{Let } v = (G_k - A)^{\frac{1}{2}} u
\end{align}
$$


而贪心方法每次在一组基 $v \in [e_i]$ 中选取使得 $\tau_A(G)$ 下降最快的方向，也即，


$$
\begin{align}
\max_u tr(\frac{u^\top (G_k-A)^2 u}{u^\top (G_k - A) u}) = \max_v tr(\frac{v^\top (G_k -A) v}{v^\top v}) 
\end{align}
$$


此时根据更新的性质，有，


$$
\begin{align}
rank(G_k - A) \le n- k
\end{align}
$$


也即矩阵 $G_k -A$ 的rank每次递减1，那么在 $n$ 次迭代之后，将有，


$$
G_k  = A
$$


也即在第 $n$ 次迭代后近似的矩阵一定找到了精确解，



此时可以得到 $\tau_A(G_k)$ 的递推关系满足，


$$
\begin{align}
\tau_A(G_{k+1}) &= \tau_A(G_k) - \max_u \frac{u^\top (G_k-A)^2 u}{u^\top (G_k - A) u} \\
&=  \tau_A(G_k) - \max_{e_i} \frac{e_i^\top (G_k - A) e_i}{e_i^\top e_i} \\
&= \tau_A(G_k) - \max_{e_i} e_i^\top (G_k - A) e_i \\
&\le \tau_AG_k) - \frac{1}{n-k} tr(G_k  -A) \\
&= (\frac{n-k-1}{n-k}) \tau_A(G_k)
\end{align}
$$


递推可以得到，


$$
\begin{align}
\tau_A(G_k) \le \frac{n-k}{n} \tau_A(G_0)
\end{align}
$$

**Random Version**



对于随机算法，每次在标准多元正态中抽样一个方向 $u_i$ 作为算法的更新方向，利用矩阵的谱分解以及正交变换下仍为标准正态，


$$
\begin{align}
\mathbb{E}[\tau_A(G_{k+1})] &= \tau_A(G_k) - \mathbb{E}_u [\frac{u^\top (G_k - A)^2 u}{u^\top (G_k - A) u}] \\
&= \tau_A(G_k) - \mathbb{E} [ \frac{v^\top (G_k - A) v}{v^\top v}] ,\text{Let } v = (G_k - A)^{\frac{1}{2}} u \\
&= \tau_A(G_k) - \mathbb{E} [ \frac{\sum_{i=1}^{n-k}\lambda_i \tilde v_i^2 }{\sum_{i=1}^{n-k} \tilde v_i^2 }], \text{Let } G_k - A = Q \Lambda Q^\top, \tilde v = Q v \\
&= \tau_A(G_k) - \sum_{i=1}^{n-k} \lambda_i \mathbb{E}[ \frac{\tilde v_i^2}{\sum_{j=1}^{n-k} \tilde v_j^2}] \\
&=\tau_A(G_k) - \frac{1}{n-k}tr(G_k - A) \\
&= (\frac{n-k-1}{n-k}) \tau_A(G_k)
\end{align}
$$




因此对于随机版本的算法也具有相同的收敛率，



### DFP





对于DFP算法，也可以应用类似地分析，


$$
\begin{align}
\sigma_A(G_{k+1}) &= \sigma_A(G_k) - 2 \frac{u^\top G_k u}{u^\top A u} +(\frac{u^\top G _k u }{u^\top A u}+1) = \sigma_A(G_k) - \frac{u^\top G_k u}{u^\top A u} +1 \\
\end{align}
$$

**Greedy Version**



对于贪心算法，同样对矩阵进行分解和变换后，在一组基 $e_i$ 中选择最速下降的方向，


$$
\begin{align}
\sigma_A(G_{k+1}) &=  \sigma_A(G_k) - \max_u \frac{u^\top G_k u}{u^\top A u} +1 \\
&= \sigma_A(G_k ) - \max_{e_i} \frac{e_i^\top A^{-\frac{1}{2}}G_k A^{-\frac{1}{2}} e_i}{e_i^\top e_i}+1 \\
&= \sigma_A(G_k) - \frac{1}{n} tr(A^{-1} G_k) +1\\
&= (1- \frac{1}{n}) \sigma_A(G_k)
\end{align}
$$


递推可以得到，


$$
\begin{align}
\tau_A(G_k) \le \frac{n-k}{n} \tau_A(G_0)
\end{align}
$$



**Random Version**



对于随机算法，抽样标准正态随机向量 $v$ 可以得到期望下相同的收敛率结果， 


$$
\begin{align}
\sigma_A(G_{k+1}) &= \sigma_A(G_k) - \mathbb{E}_u [\frac{u^\top G_k u}{u^\top A u}] +1 \\
&= \sigma_A(G_k) - \mathbb{E}_v[\frac{v^\top A^{-\frac{1}{2}} G_k A^{-\frac{1}{2}}v}{v^\top v}] +1\\
&= \sigma_A(G_k) - tr(A^{-1} G_k) \mathbb{E}_v [\frac{vv^\top}{v^\top v}] +1 \\
&= \sigma_A(G_k) - \frac{1}{n} tr(A^{-1} G_k) +1\\
&= (1- \frac{1}{n}) \sigma_A(G_k)
\end{align}
$$


## Unified Analysis of Broyd Class



利用Broyd族，可以将上述三个算法用统一的框架进行分析，


$$
\begin{align}
G_{k+1}  =\text{Broyd}_\tau(G_k,A , u) &= \tau \text{DFP}(G_k,A,u) + (1- \tau) \text{SR1}(G_k,A,u) \\
&= \tau [G_k -\frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u } + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top A u} ] \\
&\quad +(1- \tau) [G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u}]
\end{align}
$$


可以证明，BFGS也属于Broyd族，只需要令 $\tau = \frac{u^\top A u}{ u^\top G u}$, 则


$$
\begin{align}
G_{k+1} &= \text{Broyd}_\tau(G_k,A , u) \\
&=  \tau \text{DFP}(G_k,A,u) + (1- \tau) \text{SR1}(G_k,A,u) \\
&= G_k  -  \frac{A uu^\top G_k + G_k uu^\top A}{u^\top G_k u } + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top G_k u} - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top G_k u} \\
&= G_k - \frac{G_k uu^\top G_k}{u^\top G_k u }  + \frac{A uu^\top A}{u^\top A u} \\
&= \text{BFGS}(G_k,A,\mu)
\end{align}
$$



重要的性质是 Broyd族关于参数 $\tau$ 满足单调性，



$$
\begin{align}
\text{Broyd}_\tau(G_k,A , u)
&= \tau [G_k -\frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u } + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top A u} ] +(1- \tau) [G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u}] \\
&= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\
&\quad+ \tau[\frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top A u} -  \frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u }] \\
&= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\ \\
&\quad + \tau [\frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} + (\frac{u^\top (G_k-A) u}{u^\top A u}+2) \frac{A uu^\top A}{u^\top A u} -  \frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u }] \\
&= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\ 
&\quad + \tau [\frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} + (\frac{u^\top (G_k-A) u}{u^\top A u}) \frac{A uu^\top A}{u^\top A u} -  \frac{A uu^\top (G_k-A) + (G_k-A) uu^\top A}{u^\top A u }] \\
&= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\ 
&\quad + \tau u^\top(G_k - A) u [\frac{(G_k - A) uu^\top(G_k -A)}{(u^\top(G_k - A) u)^2} + \frac{A uu^\top A}{(u^\top A u)^2} -  \frac{A uu^\top (G_k-A) + (G_k-A) uu^\top A}{(u^\top A u) (u^\top (G_k-A) u)  }] \\
&= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\ 
&\quad + \tau u^\top (G_k - A) u ss^\top, \text{ with } s  = \frac{(G_k -A) u}{u^\top(G_k -A) u} - \frac{Au}{u^\top A u}
\end{align}
$$



因此，成立，


$$
\begin{align}
\text{SR1}(G_k,A, u) \preceq \text{BFGS}(G_k ,A, u) \preceq \text{DFP}(G_k ,A, u)
\end{align}
$$




并且如果存在 $\eta$ 使得下式成立，可以证明Broyd类经过迭代后下式仍然成立，


$$
\begin{align}
\text{If } A &\preceq G_k  \preceq \eta A \\
\text{Then } A &\preceq \text{Broyd}_\tau(G_k,A , u) \preceq \eta A \\
\end{align}
$$


根据单调性只只需要对Broyd的两端进行证明即可，


$$
\begin{align}
\text{SR1}(G_k,A, u) &= G_k - \frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u} \\
&= G_k -A -\frac{(G_k - A) uu^\top(G_k -A)}{u^\top(G_k - A) u}  +A\\
&=(I_n- \frac{(G_k - A)uu^\top}{u^\top (G_k - A)u}) (G_k - A) (I_n- \frac{uu^\top(G_k - A)}{u^\top (G_k - A) u}) +A \\
&\succeq A \\
\text{DFP}(G_k ,A , u) &= G_k -\frac{A uu^\top G_k + G_k uu^\top A}{u^\top A u } + (\frac{u^\top G_k u}{u^\top A u}+1) \frac{A uu^\top A}{u^\top A u} \\
&= (I_n - \frac{Auu^\top}{u^\top A u}) G_k (I_n - \frac{uu^\top A}{u^\top A u}) + \frac{A uu^\top A}{u^\top A u } \\
&\preceq \eta (I_n - \frac{Auu^\top}{u^\top A u}) A (I_n - \frac{uu^\top A}{u^\top A u}) + \frac{A uu^\top A}{u^\top A u } \\
&= \eta A - (\eta -1 ) \frac{A uu^\top A}{u^\top A u } \\
&\preceq  \eta A
\end{align}
$$

### Quadratic Minimization



本节考虑运用贪心或者随机的伪Newton方法优化二次函数，


$$
\min f(x ) = \frac{1}{2} x^\top A x - b^\top x 
$$


算法如下，


$$
\begin{align}
x_{k+1} &= x_k - G_k^{-1} \nabla f(x_k) \\
G_{k+1} &= \text{Broyd}_\tau (G_k,A,u)
\end{align}
$$


考虑梯度的 $A$ 范数的对偶范数，其收敛性等价于残差的收敛性


$$
\begin{align}
f(x) - f(x_{\ast}) &= \frac{1}{2} x^\top A x  + \frac{1}{2} x_{\ast}^\top A x_{\ast} - x^\top A x_{\ast} = \frac{1}{2} (x- x_{\ast} ) A (x- x_{\ast}) = \frac{1}{2} \Vert x - x_{\ast} \Vert_A
\end{align}
$$


并且只要找到序列 $\eta_k$ 作为近似矩阵 $A$ 的界，则可以用 $\eta_k$ 控制残差的收敛，根据


$$
\begin{align}
A &\preceq G_k \preceq \eta_k A \\
\frac{1}{\eta_k}  &\preceq  G_k^{-1} A \preceq 1 \\
\end{align}
$$


因此，


$$
\begin{align}
\Vert x_{k+1} - x_{\ast} \Vert_A &= \Vert x_k - G_k^{-1} \nabla f(x_k) - x_{\ast} \Vert_A \\
&= \Vert x_k - x_{\ast} - G_k^{-1}A(x-x_{\ast}) \Vert_A \\
&=\Vert (I- G_k^{-1} A)(x_k - x_{\ast}) \Vert_A \\
&\le (1 - \frac{1}{\eta_k}) \Vert x_k - x_{\ast} \Vert_A \\
&\le (\eta_k - 1) \Vert x_k - x_{\ast} \Vert_A
\end{align}
$$

对于BFGS和DFP算法，实际上 $\sigma_A$ 充当着的正是 $\eta_k-1$ 的角色，令 $\eta_k - 1 = \sigma_A(G_k)$, 则有，



$$
\begin{align}
O &\preceq (G_k -A) A^{-1} \preceq tr(G_k - A) A^{-1})\\
O &\preceq (G_k -A) A^{-1} \preceq \eta_k - 1 \\
O &\preceq G_k -A \preceq (\eta_k - 1)  A \\
A &\preceq G_k \preceq \eta_k A \\
\end{align}
$$




也即对于BFGS算法和DFP算法，


$$
\begin{align}
\frac{\Vert x_{k+1} - x_{\ast} \Vert_A}{\Vert x_k - x_{\ast} \Vert_A} \le \sigma_A(G_k) \le (1- \frac{1}{n})^k \sigma_A(G_0)
\end{align}
$$


而对于SR1算法，


$$
\begin{align}
\frac{\Vert x_{k+1} - x_{\ast} \Vert_A}{\Vert x_k - x_{\ast} \Vert_A} \le \sigma_A(G_k) \le \frac{\tau_A(G_k)}{\mu} \le  (1- \frac{k}{n}) \frac{\tau_A(G_0)}{\mu}
\end{align}
$$



可以发现对于二次函数的优化问题，上述三种算法都是超线性收敛的，

对于满足 $L$- 光滑的函数，选取 $G_0 = L I_n$  则算法的第一步迭代和梯度下降方法相同，此时算法满足，



$$
\begin{align}
A \preceq G_k \preceq \frac{L}{\mu} A
\end{align}
$$

此时根据序列的递推关系也可以显示写出关于收敛速率的显式估计，文章还将结果推广到除了二次函数以外的更为广义的函数上。
