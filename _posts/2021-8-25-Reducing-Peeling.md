---
title: 'Reducing and Peeling'
toc: true
excerpt_separator: <!--more-->
tags:

  - 图算法
  - 独立集
---



论文阅读笔记：[Computing A Near-Maximum Independent Set in LinearTime by Reducing-Peeling](https://dl.acm.org/doi/abs/10.1145/3035918.3035939)

<!--more-->

## Abstract

文章旨在解决图上的NP难问题：最大独立集问题(Maximun Independent Set,MIS)，针对该问题提出Reducing-Peeling框架，其中Reduce操作为根据度数较小的结点（0、1、2）的去除一定在或者一定不在最大独立集$\alpha(G)$ 中的结点从而减小图$G$的规模，Peeling操作为根据启发式去除度数较大的结点，从而在使得解接近最优解的同时使得该问题更易于解决。

文章首先提出BDOne和BDTwo两个Baseline，分别基于degree-one reducing和degree-two reducing, 其中BDOne为线性时间算法，更获得的最大独立集相对较小，而BDTwo获得的最大独立集更优，但在时间复杂度上却较大。在这两个Baseline的基础上，文章提出LinearTime算法，将degree-two vertex reducing更改为degree-two path reducing，从而保证线性时间复杂度。同时，提出NearLinear算法，基于图上结点的dominant关系，在接近线性时间下获得质量更优的解。



## Reducing-Peeling Framework

### Reducing

#### Degree-one Reduction

对于度数为1的结点$u$，其唯一邻居为$v$, 则$u$一定在最大独立集$\alpha(G)$里面，因此$v$一定不在$\alpha(G)$里面。

只需证明对于$u$不在$\alpha(G)$的情况下，让$u$在$\alpha(G)$中也不会更差，证明：

首先，此时$v$必在$\alpha(G)$中，否则让$u$加入$\alpha(G)$会获得更优的解。

所以只需考虑$v$在$\alpha(G)$中的情况，则此时$v$的所有邻居$N(v)$一定不在$\alpha(G)$中，则可以将$u,v$互换，并不会改变$\alpha(G)$的大小。

因此，可以结点$v$一定不在$\alpha(G)$中，可以将结点$v$删除。



#### Degree-two Reduction

对于度数为2的结点$u$，考虑其两个邻居$v,w$, 考虑两种情况：

* 情况1，$v,w$之间连边
* 情况2，$v,w$之间不连边

---

对于情况1，将$v,w$删去，我们需要证明，$v,w$一定不在$\alpha(G)$中。

由于$v,w$之间连边，其最多有一个结点在$\alpha(G)$中，不妨设该结点为$v$，由于此时$w$不在$\alpha(G)$中，此时情况和Degree-onw Reduction中的情况没有本质区别，同理可以将$u,v$互换，并不会改变$\alpha(G)$的大小。

因此，可以将$v,w$删去。

---

对于情况2，我们将$u,v,w$合成一个新的结点$uvw$，结点$vuw$所连接的边即为$v,w$连接的边，且$\alpha(G)$的大小增加1。

相较于$v,w$之间连边的情况，我们只需要多考虑一种情况，即$v,w$都在$\alpha(G)$中，此时$u$不在$\alpha(G)$中，此时与新的结点$uvw$在$\alpha(G)$的情况等价，因为此时旧结点$v,w$（也相当于新结点$uvw$）的邻居均不能在$\alpha(G)$中。

而其他情况与$v,w$之间连边的情况相同。此时与新的结点$vuw$不在$\alpha(G)$中的情况等价。



### Peeling

上述的Reducing操作不可能一直进行下去，当没有结点可以Reducing的时候，算法进行Peeling操作。

Peeling操作即去除度数最大的结点，理由如下：

* 度数最大的结点邻居数最多，将该结点加入$\alpha(G)$中，将导致所有的邻居都不能加入$\alpha(G)$中
* 对度数最大的结点做Peeling操作，将导致最多的结点的度数变化，使得Reducing操作更有可能进行

### Upper Bound

Reducing-Peeling框架下的所有算法尽管不能获得最大独立集MIS的精确解，但可以给出该精确解的上界。

记Reducing-Peeling得到的独立集大小为$I$，被Peeling掉的不在$I$中的结点个数为$R$,则其最优解的上界由$I+R$给出。

证明的关键在于证明每一次Peeling掉不在$I$中的结点，至多造成解距离最优解大小1的代价，所以造成的总代价为$R$.

只需考虑被Peeling掉的结点本应包含在最优解但却被Peeling掉的情况，而若将其包含在最优解中，则该节点的所有邻居都不能被包含在答案中，而Peeling掉该结点尽管对该结点本身造成了影响（至多造成1的代价），但却使得其他邻居结点具有更大的自由（其他结点不会获得更差的解，不会造成额外的代价），因此造成的总代价也至多为$R$.

## Baseline：BDOne and BDTwo

在Reducing-Peeling框架下，提出两个Baseline，流程如下：

迭代地进行Reducing操作，直到没有结点可以进行Reducing，之后进行Peeling直到Reducing操作可以继续进行。

BDOne在仅仅进行degree-one reduction，因此为线性算法。

BDTwo同时进行degree-one reduction和degree-two reduction，而degree-two reduction中可以会涉及到将结点合成一个新的结点，此时需要考虑被合成的两个结点的度数，此时每一次degree-two reduction的时间复杂度为$O(d_v+d_w)$，因此总体的复杂度为非线性的复杂度。并且文章构造出一个例子，说明该复杂度的下界不小于$\Omega(m+n \log n)$,详见文章。



## LinearTime

BDTwo的非线性时间复杂度来自于degree-two reduction，更改degree-two reduction为degree-two path reduction即可使得复杂度为线性复杂度。

---

Degree-two path指的是图$G$中的一条最大路径，这条路径上所有结点的度数均为2。而路径的两个端点记作$v,w$, 根据$v,w$的情况以及路径长度的奇偶性，可以分出四种情况：

* 情况1，$v=w$，其他下面四种情况均为$v \ne w$

* 情况2，$v,w$之间连边，路径长度为奇
* 情况3，$v,w$之间不连边，路径长度为奇
* 情况4，$v,w$之间连边，路径长度为偶

上述四种情况，都可以进行相应的Reduction操作。

---

对于情况1，可以去掉结点$v(w)$，只需证明去掉该结点一定不会比不去掉更差，首先去掉该节点后可以将degree-two path中在$\alpha(G)$中和不在$\alpha(G)$中的结点全部互换，此时所得到的独立集不仅大小不会比原先的小，而且由于结点$v(w)$被去掉之后，其邻居具有了更多的选择空间。

对于情况2，可以将结点$v,w$都去掉，因此这样可以使得path中两个端点均在在$\alpha(G)$中。而就算保留了其中的某个结点在在$\alpha(G)$中，此时会使得path只有一个端点能够在在$\alpha(G)$中，并不能增加在$\alpha(G)$的大小，反而限制了结点$v,w$的邻居的选择空间。

对于情况3，设结点$v$的邻居为$z$，可以在$w,z$之间连一条边后去掉path上的其他结点。当情况3与情况2相同的时候，$v,w$均不在在$\alpha(G)$中，而此时$z$在$\alpha(G)$中；当情况3不与情况2相同的时候，也即结点$v,w$中有一个结点在在$\alpha(G)$中，由于对称性不妨设结点$v$在$\alpha(G)$中，则此时结点$z$必然不在$\alpha(G)$中。综上，$w,z$必然不能同时在$\alpha(G)$中，此时可以将这两个结点之间连一条边，去掉path上的其他结点，并且相应地增加$\alpha(G)$的大小（与路径长度相关）。

对于情况4，$v,w$至少有一个结点在$\alpha(G)$中，因此在这两个结点之间连一条边（如果原先没有），去掉path上的其他结点。证明和情况2和情况3地证明类似，讨论即可，区别仅在路径的长度从奇数变成了偶数。

---

对于寻找degree-two path来说，每寻找到$n$个在path上的结点，就可以reduce掉$O(n)$个结点。因此该算法为线性时间的算法。



## NearLinear

NearLinear算法在LinearTime算法的基础上，基于图上结点的dominant关系，在接近线性时间下获得质量更优的解。

### Dominance Reduction

若结点$u,v$之间有边相连，且结点$u$的所有邻居也为结点$v$的邻居，则可以将结点$v$去掉。

证明较为直观，只需要考虑结点$v$在MIS中的情况，但此时将$u,v$互换，则对MIS的大小没有影响，而由于结点$u$的邻居被结点$v$的邻居包含，使用结点$u$而非结点$v$将给其他结点带来更多的选择空间。因此，该dominance reduction过后的结果并不会比之前更差。

### Triangle Counting

结点$u,v$是否可以进行dominance reduction，实际上取决于$d_u,d_v,\delta$, 其中$\delta$为包含边$(u,v)$的三角形个数。而$d_u,d_v,\delta$都可以在Reducing或者Peeling的时候在$O(1)$时间内维护。所以该算法的复杂度在于初始进行Triangle Counting的复杂度，为$O(m d)$;若将度数从小到大排序，则复杂度为$O(\sum_{(u,v) \in E} \min\{d_u, d_v \})$

若将$d$视为较小的常数，则该算法几乎是线性时间的算法。



## Accelerating ARW

文章同时还提到该Reducing-Peeling方法可以提升一个称为ARW的算法的性能，由于并非主体内容，此处暂略。



