---
title: 'Smooth Nash Equilibria'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - 博弈论
---



Paper Reading: Smooth Nash Equilibria: Algorithms and Complexity.



<!--more-->



## Introduction

考虑博弈论中经典的 $m$ 玩家博弈问题，每个玩家有 $n$ 个动作，对应的效用函数为 $A_1,\cdots,A_m : [n]^m \rightarrow [0,1]$.

给定策略$x = (x_1,\cdots,x_m) \in (\Delta_n)^m$ ，对应的效用为 



$$
\begin{align*}
A_j(x) = A_j(x_1,\cdots,x_m) = \mathbb{E}_{a_1 \sim x_1} \cdots \mathbb{E}_{a_m \sim x_m} [A_j(a_1,\cdots,a_m)].
\end{align*}
$$



经典的 $\epsilon$-Nash均衡点的定义为



$$
\begin{align*}
\max_{w \in \Delta_n} A_j(w,x_{-j}) - A_j(x) \le \epsilon
\end{align*}
$$



即使对于双玩家的非零和博弈问题，寻找一个$\epsilon$-Nash均衡点也是一个很难的问题。文章 Near-Optimal Communication Lower Bounds for Approximate Nash Equilibria中给出了一个该问题的复杂度下界证明了，存在某个常数 $\epsilon>0$ 使得寻找一个 $\epsilon$-Nash均衡点至少需要 $\Omega(n^{2-o(1}))$ 次对于效用函数 $A$的query，当决策空间 $n$ 很大的时候该复杂度可能是难以接受的。



而本文的贡献使用smoothed analysis，引入了 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点的定义，作为传统Nash均衡点的松驰，并且证明当 $m,\sigma,\epsilon$ 为常数的时候，存在常数复杂度的算法寻找到引入了 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点。其严格定义由如下的点 $x \in (\Delta_n)^m$ 给出，



$$
\begin{align*}
\max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_{-j}) - A_j(x) \le \epsilon, \text{  where  } \mathcal{K}_{\sigma,n} = \left \{ x \in \Delta^n : 0 \le x_i\le \frac{1}{n \sigma} \right\}.
\end{align*}
$$


当 $\sigma= 1/n$ 时就复原了 $\epsilon$-Nash均衡点的定义，而当 $\sigma=1$ 时所关注点为均匀分布而不具有实际意义，本文关注 $1/n < \sigma <1$ 的情况。

 

本文尝试复现原论文的该部分证明，下面的小标题与原论文的引理/定理相对应



## Lemma 3.2

根据Nash均衡点的存在性定理，存在 $x \in (\mathcal{K}_{\sigma,n})^m $ 满足 $x$ 为决策空间均为 $\mathcal{K}_{\sigma,n}$ 的博弈的0-Nash均衡点，即



$$
\begin{align*}
\max_{w \in \mathcal{K}_{\sigma,n}} A_j(w,x_{-j})  = A_j(x).
\end{align*}
$$



给定该点的存在性，我们可以采用采样的方式逼近，对于足够大的 $k$, 根据 $x_j$ 的分布采 $t$ 个样本并且取其均值 $\hat X_j$ 该随机变量作为估计

定义 $A_j(\hat X_1,\cdots,\hat X_m) = f: [n]^{km} \rightarrow [0,1]$, 由于每次该函数的输入对 $k$ 做了平均，因此对输入的每个元素的改变至多对函数值造成 $1/k$ 的改变量。据此使用 [McDiarmid's inequality](https://en.wikipedia.org/wiki/McDiarmid%27s_inequality),  给定 $i \in [n], j \in [m]$, 我们知道对于任意的 $\lambda>0$ 成立


$$
\begin{align*} 
\mathbb{E}[ \exp(\lambda (A_j(e_i,\hat X_{-j})- A_j(e_i,x_{-j})))] &\le \exp( \lambda^2 m / (2k)).
\end{align*}
$$


根据上面的第一个不等式，类似于finite class lemma，可以知道


$$
\begin{align*}
&\quad \mathbb{P} \left( \left \vert \max_{w \in \mathcal{K}_{\sigma,n} } A_j(w,\hat X_j) - \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_j) \right \vert >t  \right ) \\
&\le \mathbb{P} \left(\max_{w \in \mathcal{K}_{\sigma,n}} \vert A_j(w, \hat X_j) - A_j(w,x_j) \vert >t \right) \\
&\le \mathbb{P} \left(\exp( \lambda\max_{w \in \mathcal{K}_{\sigma,n}} \vert A_j(w, \hat X_j) - A_j(w,x_j) \vert) >\exp(\lambda t) \right) \\
&\le {\exp(-\lambda t)} \mathbb{E} \max_{w \in \mathcal{K}_{\sigma,n}} \exp (\lambda \vert A_j(w, \hat X_j) - A_j(w,x_j) \vert)  \\
&\le {\exp(-\lambda t)} \mathbb{E} \max_{w \in \mathcal{K}_{\sigma,n}} \sum_{i=1}^n w_i \exp (\lambda \vert A_j(e_i, \hat X_j) - A_j(e_i,x_j) \vert) \\
&\le   \frac{\mathbb{E} \exp (\lambda \vert A_j(e_i, \hat X_j) - A_j(e_i,x_j) \vert)}{ \sigma \exp(\lambda t)} \\
&\le \frac{1}{\sigma} \exp \left( -\lambda t + \frac{\lambda^2 m}{2k} \right) \\
&\overset{\lambda = tk /m}{\le} \frac{1}{\sigma} \exp \left(  - \frac{t^2 k}{2m}\right)
\end{align*}
$$


这就意味着，给定 $j \in [m]$, 以概率 $1-\delta$ 成立


$$
\begin{align*}
 \left \vert \max_{w \in \mathcal{K}_{\sigma,n} } A_j(w,\hat X_j) - \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_j) \right \vert \le \sqrt{\frac{2m}{k} \log \left( \frac{1}{\sigma \delta}\right)} . 
\end{align*}
$$


对于所有的 $j \in [m]$ 取联合界，可以得到以概率 $1-\delta$ 成立，


$$
\begin{align*}
 \left \vert \max_{w \in \mathcal{K}_{\sigma,n} } A_j(w,\hat X_j) - \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_j) \right \vert \le  \sqrt{\frac{2m}{k} \log \left( \frac{m}{\sigma \delta}\right)} , \quad \forall j \in [m].
\end{align*}
$$


类似地，使用 [McDiarmid's inequality](https://en.wikipedia.org/wiki/McDiarmid%27s_inequality),  给定 $i \in [n], j \in [m]$, 我们知道对于任意的 $\lambda>0$ 成立


$$
\begin{align*} 
\mathbb{E}[ \exp(\lambda (A_j(\hat X_{j})- A_j(x_{-j})))] &\le \exp( \lambda^2 m / (2k))
\end{align*}
$$


采用相同的技术，可以证明以概率 $1-\delta$ 成立，


$$
\begin{align*}
\vert A_j(\hat X_j) -A_j(x_j) \vert \le  \sqrt{\frac{2m}{k} \log \left( \frac{m}{\sigma \delta}\right)} ,\quad \forall j \in [m]. 
\end{align*}
$$


换句话说，给定任意的 $\epsilon>0$, 只需要取 


$$
\begin{align*}
k \gtrsim  \frac{m}{\epsilon^2} \log \left( \frac{m}{\sigma \delta} \right)
\end{align*}
$$


那么根据 $x_j$ 采样得到的 $\hat X_j$, 可以以概率 $1-2\delta$ 满足，


$$
\begin{align*}
\max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, \hat X_{-j}) - A_j(\hat X) \le \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w,x_{-j}) - A_j(x) + \epsilon \le \epsilon, \quad \forall j \in [m].
\end{align*}
$$


由于我们在连续的空间内寻找Nash均衡点，这个引理允许我们在一个 $\epsilon$-网格中遍历的寻找，越大的 $k$ 值对应了越小的 $\epsilon$.

这个引理自然地引出一个朴素的算法，给定 $\epsilon$, 根据上述引理设定对应的$k$, 那么每个玩家的搜索空间从无限维的空间离散化为了 $n^k$ 个取值，对于所有的$m$ 个玩家，总的搜索空间总共只有 $n^{mk}$ 种组合，我们可以遍历所有的这些点，逐一判断其是否为  $\epsilon$-近似-$\sigma$-光滑-Nash均衡点。 注意到 $\max_{w \in \mathcal{K}_{\sigma,n}} A_j(w,x_{-j})$ 实际上等价于前 $1/(n\sigma)$ 大的所有动作的效用函数的平均值，因此上述的判断可以在 $n$ 次对于效用函数 $A$ 的query中完成，最终的复杂度为 $\tilde{\mathcal{O}}(n^{m^2 \epsilon^{-2}+1})$. 这个复杂度对应于原文的Theorem 5.1中给出的结论。



然而上述的复杂度当决策空间 $n$ 很大的时候并不可接受，上述的复杂度中关于 $n$ 的依赖来源于两个方面，第一个方面是每个玩家离散化后的决策空间为 $n^k$仍然指数级依赖于 $n$, 第二个方面是在判断是否为  $\epsilon$-近似-$\sigma$-光滑-Nash均衡点 的时候必须计算所有的 $n$ 个动作排序后得到其对应的  $\max_{w \in \mathcal{K}_{\sigma,n}} A_j(w,x_{-j})$ 的效用值，这需要 $\Omega(n)$ 的复杂度。

下面我们分别解决上述的这两个问题，这也构成了本文的主要贡献。也即证明了对于寻找  $\epsilon$-近似-$\sigma$-光滑-Nash均衡点的问题，实际上的复杂度可以不依赖于决策空间的大小 $n$。这主要由于 $\sigma$-光滑化带来的好处。



## Lemma 4.2



首先，我们证明给定 $\epsilon>0$, 存在$K$ 使得可以通过在$[n]$ 中有放回的采样 $K$ 个样本点计算对应的样本均值 $\hat X_j $, 以高概率在 ${\rm supp}(\hat X) = {\rm supp}(\hat X_1) \times \cdots {\rm supp} (\hat X_m)$ 中存在一个$\epsilon$-近似-$\sigma$-光滑-Nash均衡点. 上述结论实际上是Smoothed Analysis with Adaptive Adversaries 中的Theorem 2.1 的直接推论. 沿用上一节的分析思路，令 $x \in (\mathcal{K}_{\sigma,n})^m $ 满足 $x$ 为决策空间均为 $\mathcal{K}_{\sigma,n}$ 的博弈的0-Nash均衡点，然后根据Lemma 3.2 我们可以选择 $k$ 然后对于每一个 $j \in [m]$ 都采样对应的 $B_{1,j}, \cdots B_{k,j} \sim x_j$ 然后计算其样本均值$\hat X_j$ 就可以以概率 $1-\delta$得到一个-近似-$\sigma$-光滑-Nash均衡点。记这个从 $x$ 采样的大小为$k$的样本集合为$S$， 根据Smoothed Analysis with Adaptive Adversaries文中的结论，存在一个coupling给出一个从均匀分布采样的大小为 $K = k / \sigma \log (k / \delta)$  的样本集合 $S' $ 以 $1-\delta$ 的概率包含集合 $S$. 对所有的 $j \in [m]$ 取联合概率界，那么根据这个coupling，取 


$$
\begin{align*}
K \gtrsim  \frac{k}{\sigma} \log \left( \frac{m k}{\delta} \right).
\end{align*}
$$


以概率 $1-2\delta$ 上述的采样方式包含了一个 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点. 这个引理给出的提升是显著的，原来每个玩家的搜索空间的大小为 $n^k$, 而这个引理可以将搜索空间缩小为 $K^k$, 那么总共$m$个玩家的搜索空间为 $K^{km}$. 这个缩减由于couping给出的 $S'$ 是一个均匀分布，因此只要根据这个均匀分布采样 $K$ 次得到的集合就高概率包含了所关注的 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点。



## Lemma B.2



本节的目的在于去除在测试一个给定的策略 $x$是否为 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点中复杂度关于 $n$ 的依赖，给定$j \in [m]$ 以及需要判断的策略 $x$, 最朴素的算法是遍历所有的$w \in \Delta_n$  的所有$n$ 个动作并且选取前 $n\sigma$ 大的值，然后判断是否与 $A_j(x)$ 的值差小于 $\epsilon$. 本引理主要证明了可以通过选择一个从$[n]$ 中的均匀分布中独立采样的大小为 $N$ 的集合 $R$ ，然后选取前 $N \sigma$ 大的值作为上述的估计，而 $N$ 的取值尽管依赖于 $\epsilon,\sigma$ 但是却独立于决策空间的大小 $n$， 最终给出了不依赖于 $n$ 的复杂度上界。



给定玩家 $j \in [m]$ 以及一个策略 $x \in (\Delta_n)^m$  ，定义 $\mathcal{T}(j,x)$ 为 $R$ 中在所有的 $n$ 个动作中排序前 $n\sigma$ 大的采样点所构成的集合，我们将这个集合的大小记作 $T$, 根据定义 $T \sim {\rm Bin}(N,\sigma)$, 根据Hoeffding不等式，以概率 $1-\delta$ 成立


$$
\begin{align*}
\vert T -  N \sigma \vert \le \sqrt{\frac{N \log (1/\delta)}{2}}. 
\end{align*}
$$


因此我们可以知道只要 $N \gtrsim \sqrt{\log(1/\delta)}/\sigma$ 就有 $N \sigma/2 \le T \le 3 N \sigma /2 $. 注意到 $\mathcal{T}(j,x)$ 中的点均匀地分布在前 $n\sigma$ 大的采样点所构成的集合中，对于任意给定的 $T$ 我们应用Hoeffding不等式，以概率 $1-\delta$ 成立


$$
\begin{align*}
\left \vert \frac{1}{T} \sum_{ w \in \mathcal{T}(j,x)} A_j(w, x_{-j}) - \frac{1}{n \sigma} \sum_{i=1}^{\sigma n}A_j(\pi_i (j,x),x_{-j})  \right \vert \le \sqrt{\frac{\log (1/\delta)}{2T}}. 
\end{align*}
$$


对所有的  $N \sigma/2 \le T \le 3 N \sigma /2 $ 中的 $T$ 取联合界，可以得到以概率 $1- \delta$, 成立


$$
\begin{align*}
\left \vert \frac{1}{T} \sum_{ w \in \mathcal{T}(j,x)} A_j(w, x_{-j}) - \frac{1}{n \sigma} \sum_{i=1}^{\sigma n}A_j(\pi_i (j,x),x_{-j})  \right \vert \le\sqrt{\frac{\log ( N \sigma/\delta)}{2T}}, \quad \forall N \sigma/2 \le T \le 3 N \sigma /2.
\end{align*}
$$


合起来我们知道上式以概率 $1-2\delta$ 成立，回忆我们在上一节中已经将决策空间缩小到了 $K^{km}$ 个，我们对该策略空间中的所有策略取概率联合界，就得到了给定玩家 $j \in [m]$ 对于任何一个策略 $x \in {\rm supp}(\hat X)$ 以概率 $1-\delta$ 有，


$$
\begin{align*}
&\quad \left \vert \frac{1}{T} \sum_{ w \in \mathcal{T}(j,x)} A_j(w, x_{-j}) - \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_{-j})  \right \vert \\ 
&\le \left \vert \frac{1}{T} \sum_{ w \in \mathcal{T}(j,x)} A_j(w, x_{-j}) - \frac{1}{n \sigma} \sum_{i=1}^{\sigma n}A_j(\pi_i (j,x),x_{-j})  \right \vert \\
&\lesssim\sqrt{\frac{km}{N \sigma} \log \left( \frac{KN \sigma}{\delta}\right)}, \quad \forall x \in {\rm supp}(\hat X).
\end{align*}
$$


又由于在对数意义下成立 $T \asymp N \sigma$， 而我们将 $R_j$ 中的所有元素关于效用函数的值从大到小排序后得到序列 $A_j(r_{j,\nu_1},x_{-j}) \ge \cdots A_j(r_{j,\nu_N},x_{-j})$ 以及相应的排列 $\nu$, 我们知道给定以概率 $1-\delta$ 成立


$$
\begin{align*}
&\quad\left \vert 
 \frac{1}{T} \sum_{ w \in \mathcal{T}(j,x)} A_j(w,x_{-j}) - \frac{1}{N \sigma} \sum_{i=1}^{\sigma N} A_j(r_{1,\nu_i},x_{-j})
\right \vert \\
&= \left \vert 
 \frac{1}{T} \sum_{ i =1 }^{T} A_j(r_{1,i},x_{-j}) - \frac{1}{N \sigma} \sum_{i=1}^{\sigma N} A_j(r_{1,\nu_i},x_{-j}) 
\right \vert \\
&\le \frac{2 \vert T - N \sigma \vert}{\max\{ T , N \sigma\}} \\
&\lesssim \frac{1}{\sigma}\sqrt{\frac{ km}{N } \log \left(\frac{K}{\delta} \right)}, \quad \forall x \in {\rm supp}(\hat X). 
\end{align*}
$$


合起来这告诉我们对于任意给定的 $j \in [m]$ 成立，


$$
\begin{align*}
\left \vert \frac{1}{N \sigma} \sum_{i=1}^{\sigma N} A_j(r_{1,\nu_i},x_{-j}) -  \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_{-j}) \right \vert \lesssim\sqrt{\frac{km}{N \sigma} \log \left( \frac{KN \sigma}{\delta}\right)}, \quad \forall x \in {\rm supp}(\hat X).
\end{align*}
$$

$$
\begin{align*}

\end{align*}
$$
最后我们关于所有的 $j \in [m]$ 取概率联合界，得到


$$
\begin{align*}
 \left \vert \frac{1}{N \sigma} \sum_{i=1}^{\sigma N} A_j(r_{1,\nu_i},x_{-j}) -  \max_{w \in \mathcal{K}_{\sigma,n}} A_j(w, x_{-j}) \right \vert \lesssim \frac{1}{\sigma} \sqrt{\frac{km}{N} \log \left( \frac{KN m\sigma}{\delta}\right)}, \quad \forall x \in {\rm supp}(\hat X), \forall j \in [m].
\end{align*}
$$


这也就是说我们只需要选取


$$
\begin{align*}
N \gtrsim \frac{km}{\epsilon^2 \sigma^2} \log \left( \frac{KN m\sigma}{\delta} \right)
\end{align*}
$$


那么就可以以概率 $1-\delta$ 判断出一个任何给定的策略 $x$ 是否是一个 $\epsilon$-近似-$\sigma$-光滑-Nash均衡点。

而上述的算法复杂度仅仅只有 $N$, 不依赖于决策空间的总数$n$, 至此完成了这一部分的证明，这也即原文Theorem 5.2 中所叙述的事情。

在此我们对所有的内容做一个总结，首先一个Nash均衡点是一个连续的策略，而Lemma 3.2将其用网格 $1/k,\cdots,1$ 离散化，其中 $k= \tilde{O}(m \epsilon^{-2}) $ 将总体的策略空间从无穷维降低为$n^{km}$ , Lemma 4.2 使用样本量为 $K =\tilde{\mathcal{O}}(k/\sigma)$ 的coupling近似 $n$ 维空间, 将Nash均衡点的策略的搜索空间进一步降低为 $K^{km}$. Lemma B.2 将判断给定的策略是否满足要求中计算前 $n \sigma$ 个效用函数的平均近似为计算前 $N \sigma$ 个的平均，其中 $N = \tilde{\mathcal{O}}( km   \epsilon^{-2} \sigma^{-2})$. 所以算法的总复杂度为 


$$
\begin{align*}
N K^{km} = \tilde{\mathcal{O}} \left( \frac{m}{\sigma \epsilon^2}  \right)^{\tilde{\mathcal{O}}\left(\frac{m^2}{\epsilon^2} +1 \right)}.
\end{align*}
$$


当 $\sigma,\epsilon,m$ 都为常数的时候，这给出了一个常数复杂度的随机算法，这与计算 $\epsilon$-Nash均衡点的 $\Omega(n^{2 - o(1)})$ 的复杂度形成了鲜明的对比。




















$$
\begin{align*}


\end{align*}
$$






