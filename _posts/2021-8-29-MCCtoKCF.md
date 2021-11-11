---
title: 'Maximum Clique Computation to K-Clique Finding'
toc: true
excerpt_separator: <!--more-->
tags:
  - 图算法
  - 最大团
---



论文阅读笔记：[Efficient Maximum Clique Computation over Large Sparse Graphs](https://dl.acm.org/doi/abs/10.1145/3292500.3330986)

<!--more-->

## Abstract

大多数求解MCC问题（Maximum Clique Computation）的算法都是基于稠密图的算法（图用邻接矩阵表示），但不能直接推广到用邻接表表示的稀疏图上，但实际应用中却主要面向稀释图。文章主要的贡献是证明稀疏图上的MCC问题（MCC-Sparse）等价于文章定义的一系列稠密图上的KCF问题（K-Clique Finding，KFC-Dense）。并且给出了基于Upper Bound和Reduction的两种剪枝框架，以及给出了一种接近线性时间的启发式算法。

## MCC to KCF

### K-Clique Finding

从每个结点的角度考虑，由于最大团一定由结点组成。考虑每个结点以及其邻居构成的子图(ego-graph），最大团一定在里面。且此时邻居结点之间两两连边，也即此时ego-graph的最大团为整张图的最大团。由于枚举邻居集合会有重复的现象，可以考虑给所有结点定个vertex order，在考虑ego-graph的时候只考虑vertex order（or rank）比该结点大的邻居结点。也即有：



$$
\max_{u \in V} w(N^{+}(u))+1 = w(G)
$$

其中，$w$表示最大团，$N^{+}(u)$表示比一个结点rank大的邻居结点构成的图，加上该结点本身，也即一个ego-graph。

---

上述说明图上的MCC问题可以转化为每个结点的Ego-Graph上的MCC问题，下面再说明该问题可以进一步简化为Ego-Graph上的K-Clique Finding问题，考虑从rank大的结点开始枚举，我们有如下不等式：



$$
w(N^{+}(v_i)) \le \max_{j>i}w(N^{+}(v_j))+1
$$



也即我们不断考虑当前的最大团，仅当结点$v_i$位与新的最大团中的时候，最大团的大小增加1。记之前找到的最大团大小为$k$，此时只需要寻找一个$k+1$团即可。

### Degeneracy Order：An Optimal Order

由于该算法需要执行$V$次ego-graph上团的寻找，而ego-graph通常是一个小的稠密图，因此再每个ego-graph都可以构建邻接矩阵，来调用之前用于MCC-dense上面的算法。而复杂度的上界由ego-graph的最大大小决定，因此一个好的vertex order应该使得$\max_{u \in V} \vert N^{+}(u) \vert $ 最小。文章采用的是degeneracy order，并且证明该order是最优的一个order。

---

首先给出degeneracy order的定义：

对于一张图$G$，不断移除其度数最小的结点，直到$G$空，此时的顺序也即degeneracy order。该算法其实也是$O(E)$时间内寻找k-core的算法，此时的degeneracy order按照core为第一比较级，度数为第二比较级排序。同时定义$\max_{u \in V} core(u) = \delta$, 该算法同时计算出了$\delta$.

并且有下式成立:



$$
\max_{u \in V} \vert N^{+}(u) \vert \le \delta
$$



证明：设$core(u)=k$ ,则根据degeneracy order，$N^{+}(u)$必然为k-core中的结点，此时在结点$u$与$N^{+}(u)$构成的子图中，结点$u$的度数不大于$k$，而$\delta =\max k$,得证。

---

再者，对于任意的order，由于最大的k-core中的结点一定会被涉及到在$N^{+}(u)$中，也即一定存在结点$u$,使得$\vert N^+{(u)} \vert \ge \delta$，从而有在任意的order下，都有$\max_{u \in V} \vert N^{+}(u) \vert  \ge \delta $, 因此再由上面的结论，采用degeneracy order是一个最优的order。此时每个ego-graph地大小被$\delta$所限制，而由其他文章给出的结论，可知$\delta \le \sqrt{2m+n}$，而且根据实验实际上该值通常更小，因此寻找每一个ego-graph的k-clique寻找的复杂度并不高。

## Reduction

考虑进行Reduction，由于最大团问题（MCC）和最大独立集问题（MIS）为图上的补问题，因此实际上MIS上的Reduction也可以用于最大团或者k-clique上（KCF）的Reduction。关于MIS上的Reduction，可以详见：

论文：[Computing A Near-Maximum Independent Set in LinearTime by Reducing-Peeling](https://dl.acm.org/doi/abs/10.1145/3035918.3035939)

但在本文的KCF上的Reduction中，除了在Reducing-Peeling Framework上所提到的degree-(0,1,2) reduction,加上了degree-3 reduction，其本质和处理方法是相同的，但要考虑的情况更多一些，时间复杂度也相对高一些。

## Upper Bound

对于最大团，如果可以预先找到一些上界，可以避免无效搜索。当找到的k-clique的k值已经达到上界的时候，可以提前终止算法。

### Degree-Based Bound

直观的上界来自于结点的度数，结论是显然的，不予证明：



$$
w(G) \le \max_{u \in V} d(u) +1
$$





### Core-Based Bound

由于一个k-clique必须首先是一个k-core（结点度数均不小于k的最大连通子图），因此基于k-core的$\delta$，给出了$w(G)$的另一个上界。且根据k-core的定义，可以知道Core-Based Bound是一个比Degree-Based Bound更紧的上界：



$$
w(G) \le \delta +1\le \max_{u \in V}d(u)+1
$$



### Color-Based Bound

根据团的定义，若对一个团进行图染色，至少需要最大团大小的颜色，因此图染色的最小数目必然是最大团的上界。

但求解一个图的最小染色数目是一个NP难问题，文章采用基于degeneracy order的贪心算法求解一种可能的图染色方案：也即从rank最大的结点开始依次对结点赋值（染色），每次将一个结点赋值为邻居结点都没有赋值过的最小值，该算法也是$O(E)$的。

该贪心算法不仅可以给出一个最大团大小的上界，还是一个比Core-Based Bound更紧的上界，因为在图染色时仅仅考虑ego-graph，而根据$\max_{u \in V} \vert N^{+}(u) \vert \le \delta$， 则有$color(G) \le \max_{u \in V} \vert N^{+}(u) \vert +1 \le \delta+1$。

---

因此，在上述三个上界中，基于degeneracy order的Color-Based Bound是最紧的上界，且可以在$O(E)$时间内计算得到。



## Algorithm

### Heuristic Algorithm

对于近似解，可以用启发式算法搜索：

主要有两个启发式：

* 最大团通常包含度数最大的结点
* 最大团很有可能是degeneracy order的一个后缀（称为degeneracy-based clique），比如很有可能是一个最大的k-core

因此，该启发式分为两步:

* 寻找几个度数高的结点，以其为中心寻找最大团
* 在degneracy的后缀中寻找最大团

---

可以根据degree-based bound, core-based bound, color-based bound给出最大团大小的上界，当近似算法的结果等于上界时说明找到了精确解，否则仅仅为一个叫近似解。而上述启发式算法为线性复杂度: $O(E)$.

### Exact Algorithm

在使用精确解算法之前，可以先调用代价较小的近似解，尝试是否可以找到精确解。

同时，近似解中的最大团和基于度数、core、图染色的上界等，都可以给出最大团大小的界，用于估计以及剪枝。

然后将MCC问题转化为KCF问题，经过Reduce后寻找每个ego-graph的k-clique，最后取最大值即可。

### Approximate Algorithm

有的应用场景仅需要近似解，仅仅需要放宽KCF中的计算，将对每个ego-graph的k-clique的寻找，更改为寻找一个degeneracy-based clique即可。当然，在使用Excat Algorithm之前，也可以先调用代价较小的Approximate Algorithm，也许可以预先得到精确结果，或者得到一个可能更紧的界更有利于Exact Algorithm。由于该近似算法基于为线性复杂度：$O(\delta E)$.









