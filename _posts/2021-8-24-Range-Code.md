---
title: 'Range Code'
toc: true
excerpt_separator: <!--more-->
tags:

  - 图算法

---

论文阅读笔记：[Accelerating Set Intersections over Graphs by Reducing-Merging](https://dl.acm.org/doi/pdf/10.1145/3447548.3467219)



<!--more-->



## Abstract

RangeCode旨在加速图上的邻居集合求交操作（set intersection on graph），该操作是图上一些经典算法，如：triangle counting（三角计数）, maximal clique enumeration（最大团搜索）, and subgraph matching（子图匹配）问题的基本操作。因此加速该操作对提升其他图算法的性能非常重要。文章主要有以下贡献：

* 使用ruducing-merging框架解决该操作，其中reduce即先根据集合的定义域范围（range）首先去掉一些不可能作为结果出现在交集中的结点，merge即将两个集合的相同部分合并为交集生成最终结果。
* 对于图上的结点使用不同的编码（range code）可能导致不同的结果，而编码方式直接影响reduce的性能，因此文章定义了一个最优编码问题（range code optimization），并且证明其为NP难问题。
* 针对上述的最优编码问题（range code optimization)，给出一种基于贪心思想的解法。并且对于不同的需求：全局集合求交或局部集合求交（global/local intersection），给出两种不同的算法：GlobalRange和LocalRange
* 普通算法的基础上，将该问题扩展为2-level问题，也即经过两步进行reduce-merge操作而非原算法的单步（one-level），以此加速所提出的算法。



## Reduce-Merge Framework



#### 1.1 Problem of Set Intersection

给定两个结点$u,v$, 其邻居$N(u),N(v)$ 构成了两个集合$S_a,S_b$, 求解$S=S_a \cap S_b$ .（集合元素已经有序）



#### 1.2 Naive Method

最简单的方法是遍历$S_a,S_b$中的每个元素，然后将相同的元素加入到$S$中。

该方法的时间复杂度为$O(\vert S_a \vert+\vert S_b \vert)$.

该方法即为简单的Merge步。



#### 1.3 Reduce-Merge Method

Reduce-Merge框架在Merge步之前加入Recuce步，进行简单的剪枝。

已知$S_a$中元素的取值范围为$[l_a,r_a]$,$S_b$中的为$[l_b,r_b]$, 则$S$中的集合元素取值范围必为$[l,r](l=min(l_a,l_b),r=max(r_a,r_b))$. 因此可以预先排除不在这个范围内的元素，此为Reduce。

由于集合中的元素已经预先排好序（可以off-line预先做好），Reduce可以使用二分查找，复杂度为$O(log(\vert S_a \vert) + log(\vert S_b \vert ))$.

Reduce操作之后将获得更小的$S_a',S_b'$ 集合，之后再使用Merge操作获得交集。



#### 1.4 Time Complexity

时间复杂度为$O(log(\vert S_a \vert) + log(\vert S_b \vert ) +\vert S_a' \vert+\vert S_b' \vert)$.



## Range Code Optimization



#### 2.1 Range Code Optimization 

由于Reduce操作中用到了集合元素的Range，而Range来自元素的id，因此不同的id标记方法（Range Code）会造成不同的效果。

考虑上述的Reduce操作，实际上为一种Filter操作，也即$S_a$中超过$[l_b,r_b]$的元素被Reduce了，也即$S_a$被$S_b$ 进行了Filter。同理，$S_b$也会被$S_a$进行Filter。最优的编码$f$应该最大化Filter掉的元素，也即最小化Filter后剩下的元素，最理想的情况下，$S_a'=S_b'=S$.

因此，将其定义为优化问题：


$$
\min_f L(v \vert u) + L(u \vert v)
$$


其中，$L(u \vert v)$ 表示集合$N(u)$ 被集合$N(v)$ Filter后剩下的元素个数。



#### 2.2 NP-Hard 

上述最优编码问题是NP-Hard的。

证明：将其归约为图$G$上的NP难问题，最优线性指派问题（Optimal Linear Arrangement ，OLA). 该问题定义为，寻找一个函数$f$, 使得下式取到最小值：


$$
\min_f  \sum_{(u,v) \in E} \vert f(u) - f(v) \vert
$$


下面我们证明，若Range Code Optimization Problem存在多项式时间内的解，那么OLA问题也存在多项式时间内的解：

首先对于任意图$G$，我们进行如下构造：

* 对于$G$中的每条边$(u,v)$, 新增结点$x$和边$(u,x),(v,x)$,删除边$(u,v)$, 新增的$\vert E \vert $个节点集合为$T_1$
* 对于$G$中的每个结点$u$,新增其邻居结点$x$，直到结点$u$的度数为$G$的最大度数，此时$V$中所有结点度数相同，记新增的结点集合为$T_2$.

上述构造后获得了一个新图$G'$，其结点集合为$V+T_1+T_2$.

考虑$G'$上对于集合$T=T_1+T_2$的最优编码问题。


$$
\min_f \sum_{u,v \in T} L(u \vert v) + L(v \vert u)
$$


首先，由于集合$T$中的元素互不相邻，且$V = \{N(u) \vert u \in T\}$ ，因此该最优编码问题仅取决于$V$上的编码。

再者，将$T$出发的Filter操作拆分为从$T_1,T_2$两部分出发的Filter。从$T_1$中的结点$x$出发的Filter操作，由于$x$仅有一个邻居$u$,仅有$T$中以$u$为邻居的结点才会经过Filter后被保留唯一元素$u$, 否则被Filter后应为空集。而$u$正好为$V$中的结点，而$V$中每个结点恰好有$d$个邻居，因此从$u$出发的Filter将总共贡献$d$的代价，因此从$T_1$出发的这部分Filter的代价为恒定的量$\vert T_1| d$.

因此我们仅仅考虑从$T_2$中的结点$x$出发的Filter操作，结点$x$仅有两个邻居$u,v$，不妨设$f(u) < f(v)$,$N(x)$的范围为$[f(u),f(v)]$, 因此仅有$T$中在此范围内的恰好$f(v)-f(u)+1$个结点在Filter之后剩下，而这些结点正好又属于集合$V$，因此每个结点的度数相同，今次从$T_2$出发的这部分Filter的代价为$d(f(v)-f(u)+1)$.

因此，该最优编码问题实际上为：


$$
\min_f d(\vert T_1| +\sum_{x \in T_2} (f(v)-f(u)+1) = \min_f \sum_{x \in T_2} f(v) - f(u) = \min_f \sum_{u,v \in E} \vert f(u) -f(v) \vert
$$


也即，因为$T_2$中的结点和$G$中的每一条边一一对应，因此该最优编码问题实际上和$G$上的OLA问题等价。



## Global Range Code 

#### 3.1 Divide into Part1 and Part2

全局Range Code为较简单的形式，也即求解问题:$$ \min_f \sum_{u,v \in V} L(u \vert v) + L(v \vert u)$$

对于该NP难问题，给出基于贪心的解法Global Range Code，每次对$V$中的结点$x$赋值为$h$，使得该赋值操作产生的代价最小。

可以按照一个顺序给$G$中的所有结点依次赋值，比如按照BFS序，直到所有结点都被赋值为止。

考虑每次赋值新结点$x$的代价，赋值新结点$x$仅仅影响了$x$的邻居$u$，因此从$u$的角度出发考虑：

新结点$x$可能导致以下两个独立的变化：

* Part1：$u$的邻居$N(u)$个数发生变化，使得其他集合对$N(u)$进行Filter操作后剩下的元素更多

* Part2：$u$的Range发生变化，使得更多的元素被Filter后剩下

因此，枚举所有的$h$,分别计算Part1和Part2,比较获得最优的$h$.



#### 3.2 Computation of Part1

考虑Part1，也即Range包含了$h$的结点个数，有两种可能：

* 情况1，Range本来就包含了$h$
* 情况2，Range本来不包含$h$，但因为新结点的加入，使得Range包含了$h$

对于情况1，只需要统计$h \in [l,r]$ 的结点个数

对于情况2，也即$x$的邻居中，Range不包含的$h$的结点个数, 也即其度数$d$减去$h \in [l,r]$ 的邻居结点个数(因为不包括$x$本身，再减去1)



我们发现，两种情况可以统一为为对于搜索范围内的每一个$h$, 以及一个结点集合中每一个结点的Range=$[l,r]$, 统计$h \in [l,r]$ 的元素个数，区别仅在于结点集合的大小不同。对于该问题，Naive方法可以遍历每一个$h$,再遍历集合中的每一个结点的Range，统计个数，复杂度为$O(R d)$, 其中$R$为$h$的可能取值范围，$d$为结点$x$的度数。

同时，在Part1中，对于$x$的每个邻居$u$，上述所要计算的量都是一样的,最终答案乘以$x$的度数即可:


$$
Part1 = d_x(C(x)+d_x-1-C’(h))
$$


其中，$C(x)$表示图中$h \in [l,r]$ 的结点个数（包括$h$), $C'(h)$表示满足$h \in [l,r]$ 的$x$的邻居结点个数



#### 3.3 Acceleration for Computation of Part1

使用前缀和的思想，可以减少时间复杂度，记$h \in [l,r]$ 的结点个数为$C(h)$

考虑根据$h$从小到大递推，考虑根据$C(h)$递推$C(h+1)$:

已知$C(h)$,对于一个Range=$[l,r]$的结点$u$,当且仅当$l=h$的时候，$C(h+1)$在$C(h)$的基础上增加1，当且仅当$r=h$的时候，$C(h+1)$在$C(h)$的基础上减少1，也即仅当$h$在端点处的时候对$C(h)$有影响，推导如下：


$$
C(h+1) = If(l \le h+1 \le r) = If(l \le h \le r) + If(h=l-1) -If(h=r) = C(h) + If(h=l-1) -If(h=r)
$$


使用上述递推公式，复杂度为$O(R+d)$.



#### 3.4 Boundary of h

尽管使用递推代替枚举可以减小复杂度，但当$h \in [-\infty,\infty]$ 的时候，搜索代价过大。因此，需要对$h$有一个更紧地上界：

已知当前$f$的值域范围为$[f_{min},f_{max}]$ ,则此时$h \in [f_{min}-1,f_{max}+1]$.

证明是显然的，因为当$h \le f_{min} -1$ 的时候，由于之前所有结点的取值都大于或者小于$h$，令$h$为任意值是等价的，不妨令$h=f_{min}-1$；同理，当$h \ge f_{max}+1$的时候，不妨令$h=f_{max}+1$.



#### 3.5 Computation of Part2

Part2的代价来自于$h$给结点$u$带来的Range的扩展，此时所有值在$[r+1,h] \text{ or } [h,l-1]$  中的结点都将贡献1的代价，因此只需统计该结点数即可。同样类似前缀和的思想，可以在$O(R+d)$时间内计算。

记$T(h)$为落在$[0,h]$中的结点个数，则落在范围$[r+1,h]$的结点个数为$T(h)-T(r)$。而$T(h)$可以根据$h$递推得到，对于一个结点$u$，其取值为$y$,则有: $T(h) = T(h-1) + If(y=h)$。

因此，这个情况下，Part2的计算公式为累计所有$u$产生的代价:


$$
Part2 = \sum_{u \in N(x)} T_u(h) - T_u(r)
$$
其他两种情况类似，此时从略。



## Local Range Code

Local Range Code才是真正的riangle counting（三角计数）, maximal clique enumeration（最大团搜索）, and subgraph matching（子图匹配）等问题的需求，在此类问题中，仅仅两个在图上相邻的结点需要进行集合求交操作，而非Global Range Code中图上任意两个结点对。因此在Local Range Code中，计算最终的Range Code Optimization的定义域也改为图上相邻的结点。

同样地，Local Range Code在计算上也可以拆分为Part1+Part2，并且采用前缀和优化。

此处暂略，细节详见论文。



## Two-Level Merge

之前的算法可以称为One-Level Merge，也即为每个结点重新分配一个id（Range Code），利用这个新的id进行集合求交操作，而此时的Range Code必须是不重复的结点的唯一标识。但考虑到在实际问题中，结点以及自带了一个id，那么我们只需要为结点重新分配一个label，将这个label作为Range Code，而label值可以重复。此时求交分为两个步骤：

* 步骤一，将两个label集合求交
* 步骤二，取出label在交集中的结点，再进行普通的求交操作



对于Two-Level Merge算法，之前的代价计算公式也要进行相应的更改。代价分为两部分：

* 步骤一Filter后剩下的label
* 在Label满足步骤一的结果的情况下，步骤二Filter后剩下的结点



根据上述定义的代价，可以给出Two-Level Merge的贪心方法中每一步赋值的增量代价，同理可以使用类似地前缀和优化。由于公式推导过程和结论较为复杂，详见论文。此处暂略，或者过后有时间补上。



