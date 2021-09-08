---
title: 'Near Maximum Weighted Independent Set'
toc: true
excerpt_separator: <!--more-->
tags:
  - 图算法
  - 独立集


---

论文阅读笔记: [Towards Computing a Near-Maximum Weighted Independent Seton Massive Graphs](https://dl.acm.org/doi/abs/10.1145/3447548.3467232)

<!--more-->

## Abstract

文章旨在解决图上的带权最大独立集问题（Maximun Weighted Independent Set，MWIS）。

可以看作文章 [Computing A Near-Maximum Independent Set in LinearTime by Reducing-Peeling](https://dl.acm.org/doi/abs/10.1145/3035918.3035939) 从无权图到带权图上的推广。

文章提出了以下几个部分：

* 针对度数小的结点的low-degree reduction
* 针对度数大的结点的high-degree reduction
* 带权图上的Peeling，以获得近似解



## Peeling

Peeling操作基本与MIS的Peeling操作相同，但必须考虑度数。

启发式的思想来源于：度数大或者权重小的结点出现在MWIS的可能性较低。

基于上述启发式思想可以提出三种不同的启发式策略：

* degree-oriented： Peeling掉度数最大的结点
* weight-orieted：Peeling掉权重最小的结点
* hybrid：Peeling掉邻居结点和该节点权重之差最大的结点，这样一个结点的邻居结点越多，该结点本身的权重越小，该结点悦可能被Peeling掉，也即寻找$\min_{u} \sum_{v \in N(u)} w(v)-w(u) $的结点$u$.



## Low-degree Reductions

### Degree-one Reduction

#### Base Cases

针对度数为1的结点的Reduction较为简单，但与无权情况相比较需要考虑结点的权重。设度数为1的结点为$u$，其唯一邻居为$v$.

此时仅有两种情况：

* CaseA，$u \in MWIS, v \notin MWIS$
* CaseB, $u \notin MWIS, v \in MWIS$

#### Reduction

在degree-one reduction中，考虑对以下两种情况处理：

* Case1，$w(u) \ge w(v)$, 此时毫无疑问$u$一定在MWIS中，也即仅可能出现CaseA，因此$v$一定不在。因为如果$v$在MWIS中(Case B)，可以将$u,v$互换获得更好的MWIS。对于这种情况，只需将$u,v$删去，并且将MWIS的总权重增加$w(u)$。
* Case2，$w(u) \gt w(v)$, 此时CaseA和CaseB都有可能发生。由于$v$的权重较大，$v$有可能被包含在MWIS中，此时同样将$u$删去,并且将MWIS的总权重增加$w(u)$。但$v$结点状态不确定，将其权重调整为$w(v)-w(u)$,此时$v$结点的状态可以表示CaseA or CaseB。当$v$ 结点在新图的MWIS中时，表示CaseB，否则表示CaseA。



### Degree-two Reduction

Degree-two Reduction较为复杂，设度数为2的结点为$u$, 而$x,y$为其两个邻居结点。需要考虑$x,y$之间是否连边，$u,x,y$的权重大小关系等。在Degree-two Reduction中不仅需要像Degree-one Reduction中对结点的权重进行调整以表示不同情况，还需要对结点的连边情况进行调整。该连边情况的调整基于以下的Accompany Rule。

#### Accompany Rule

对于结点$u,v$,已知$N(u) \in N(v)$, 则若$v \in WMIS$ ,可以推出$u \in WMIS$， 也即此时$u$一定伴随着$v$在WMIS中。

证明：$v \in WMIS, N(v) \notin WMIS, N(u) \notin WMIS, u \in WMIS$ 

#### Base Cases

对于此时的$u,x,y$三个结点，共有四种情况：

* CaseA，$u \in MWIS, x \notin MWIS, y \notin MWIS$
* CaseB，$u \notin MWIS, x \in MWIS, y \notin MWIS$
* CaseC，$u \notin MWIS, x \notin MWIS, y \in MWIS$
* CaseD，$u \notin MWIS, x \in MWIS, y \in MWIS$

---

下面介绍具体的Reduction操作，根据$x,y$之间的连边情况分为两类

#### Triangle-shape Reduction

$u,x,y$构成了一个三角形，也即$(x,y) \in E$的情况。

此时由于结点$x,y$不能同时存在于WMIS中，不存在CaseD的情况。

Reduction的做法是删去结点$u$，并且将结点$x,y$的权重分别调整为$w(x)-w(u),w(y)-w(u)$. （如果调整后的权重为正值的话，否则只需要将该结点($x,y$)删去，因为此时可以经过与结点$u$进行swap操作获得更优的MWIS。）

新图的MWIS和旧图的MWIS的三种情况一一对应，

* 若$x,y$都不在新图的MWIS中，则对应CaseA
* 若$x$在新图的MWIS中，则对应CaseB
* 若$y$在新图的MWIS中，则对应CaseC

#### V-shape Reduction

$u,x,y$构成了一个V形，也即$(x,y) \notin E$的情况。

不妨设$w(x) < w(y)$,考虑三种情况：

* Case1,$w(u) > w(y)$. 此时CaseB、CaseC都不如CaseA。增加一个新的结点$xy$, 满足：$N(xy) =N(x) \cup N(y),w(xy)=w(x)+w(y)-w(u)$. 当新结点$xy$在WMIS中时，对应着CaseD，否则对应于CaseA。 
* Case2, $w(x) < w(u) < w(y)$. 此时CaseB 不如CaseA，此时$y \in MWIS$的情况只能是CaseD，而CaseD中$x$伴随着$y$，因此我们调整连边表示这种伴随，$N(x) = N(x) \cup N(y)$ . 而权重调整为$w(y) = w(y) -w(u)$.

* Case3，$w(u) < w(x)$，此时CaseA-D都有可能，不能删除结点$u$。但同样可以调整MWIS的大小增加$w(u)$，但必须令$(u,x) \notin E, (u,y) \notin E, N(u) = N(x) \cup N(y),w(x)=w(x)-w(u),w(y)=w(y)-w(u)$. 此时$u$伴随着$x,y$,当$x,y$都在WMIS中时表示CaseD，此时$u$也应该包含在MWIS中以保证结果大小的一致性。



## High-degree Reduction

High-degree Reduction扩展了无权情况下的Dominance Reduction。

该Reduction操作考虑$(u,v) \in E$的时候，因此称为Single-Edge Reduction。

### Base Cases

同样先考虑Base Cases：

* CaseA，$u \in MWIS, v \notin MWIS$
* CaseB,  $u \notin MWIS, v \in MWIS$
* CaseC，$u \notin MWIS, v \notin MWIS$

### Basic Single-Edge Reduction

考虑在何种情况下，$v$可以被$u$替代而获得不会更差的MWIS，也即CaseA总不比CaseB和CaseC差。

将CaseB替换为CaseA，则将使得$N(u)-N(v)$的结点都不可能在MWIS中，至多造成$w(N(u)-N(v))$的损失。而获得了$w(u)-w(v)$的权重，因此若有:



$$
w(u) \ge w(v) +w(N(u)-W(v))
$$



则可以将结点$v$删除，此即Basic  Single-Edge Reduction。



### Extended Single-Edge Reduction

考虑在何种情况下，CaseC不可能成立。考虑将CaseC替换为CaseB，代价至多为$w(N(v)-\{ u\})$，该代价的上界为$w(N(v))-w(u)$而收益为$w(v)$,因此若：



$$
w(u) +w(v) \ge w(N(u)) \text{ or } w(N(v))
$$



则CaseA和CaseB必居其一，那么$u,v$至少有一个结点在MWIS中，此时可以删除这两个结点的共同邻居$N(u) \cap N(v)$.





