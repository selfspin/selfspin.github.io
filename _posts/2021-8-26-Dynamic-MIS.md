---
title: 'Dynamic Maximun Independent Set'
toc: true
excerpt_separator: <!--more-->
tags:

  - 图算法
  - 独立集
---



论文阅读笔记：[Efficient Computation of a Near-Maximum Independent Set over Evolving Graphs](https://ieeexplore.ieee.org/abstract/document/8509304)

<!--more-->

## Abstract

文章针对动态的最大独立集（MIS）问题，主要有以下几个贡献：

* 分析动态MIS的困难性，并且给出基于搜索的指数级复杂度的精确解法Exact Algorithm，
* 通过限制搜索的深度，并且讨论不同情况，提出度数平方级别的近似解法LSTwo，并且分析近似解法的误差上界
* 借助LazySearch的思想，避免无希望的搜索，提出线性时间的近似解法LazySearch
* 根据精确算法中成功搜索的充要条件，定义besieged/unbesieged结点，并且维护相应的数据结构进行besieged based prunning
* 提出k-petal的定义，以及相应的petal based prunning



## Exact Algorithm

### NP-Hard

首先，证明Dynamic MIS问题是NP难的。

由于MIS是NP难问题，若Dynamic MIS可以在多项式时间内解决，则可以动态地生成一张图，调用Dynamic MIS地多项式时间算法解决，从而得到该图的MIS。

矛盾，因此该问题是NP难的。



### MIS Update

Dynamic MIS的关键是，如何利用上一步的MIS结果，动态生成下一步的MIS结果。

只需要考虑如何改变某些结点的情况，使得MIS的大小尽可能不变，也即将某个路径上的结点的状态进行swap操作（原本在MIS中的变为不在MIS中，原本不在MIS中的状态变为在MIS中）

根据不同的动态操作，考虑四种情况：

* Vertex Addiction，将新增的结点加入MIS，MIS大小增加1，该情况不需要搜索，后面不考虑该情况
* Vertex Deletion，如果删除的结点原本在MIS中，必须经过搜索寻找是否存在一个swap操作，使得MIS大小不变，否则其大小减小1
* Edge Addiction，只需考虑新增边的两个结点原本都在MIS的情况，此时搜索是否对其中任意一个结点存在一个合适的swap操作

* Edge Deletion，只需考虑对于新增边的端点不在MIS的情况，可能由于删边操作导致其可以在MIS中，对两个端点考虑swap操作



### Swap Searching

阐述上述的swap操作，针对结点$u$，根据结点原本是否在MIS中，分情况考虑。

* 情况1，$u \in MIS$, swap寻找如何令$u \notin MIS$ , 且MIS大小不变
* 情况2，$u \notin MIS$, swap寻找如何令$u \in MIS$, 使得加入结点$u$之后，MIS大小增加1

容易知道，两种情况是递归定义的。

情况1成立当且仅当$u$存在一个邻居满足情况2，此时可以删除结点$u$，而对其该邻居进行swap操作。

情况2成立当且仅当$u$的所有本来在MIS中的邻居都满足情况1，因为当结点$u$加入MIS后，其所有邻居都不能在MIS中了。

上述的几种update情况中，仅有Edge Deletion进行的是情况2的swap操作，而其又可以被swap的情况1递归定义，因此下面仅考虑swap的情况1。



### Boundary

证明：每次update导致的MIS的大小变化不超过1.

对于除了Edge Deletion之外的其他情况，该结论是显然的，下面对于Edge Deletion的情况证明。

证明的关键，设被删除的边为$(u,v)$, 要证明如果结点$u$存在一个valid swap，则该swap中必然包含了结点$v$，反之亦然。

用反证法，如valid swap中不包含结点$v$，则在未删边之前进行该swap，MIS的大小增加1，与MIS的定义矛盾。

因此，就算是Edge Deletion操作，由于两个端点的swap是相互联系的，update后MIS的大小变化也不会超过1.



### Besieged State

搜索是否存在有效的swap操作时，生成了一颗搜索树（search tree）。

由于仅以情况1的swap说明，该搜索树的根节点为在MIS中的结点。

一个有效的搜索，当且仅当该搜索树的所有叶子结点均为不在MIS中的结点（定义为Besieged State）

在上述定义之下，besieged vertex代表该结点存在成功的swap，unbesieged vertex代表该结点不存在成功的swap。

---

证明：从valid swap需要保证MIS大小不变出发，也即搜索子树中MIS和non-MIS结点的数目相等，因此swap后MIS大小相等。

首先，由swap的搜索过程，我们知道valid swap所对应的搜索子树中MIS和non-MIS的结点交替出现。

再者，该搜索子树中MIS结点仅有一个后代结点，但non-MIS结点可能有多个后代结点。

因此，考虑搜索子树的每一个branch，都是以MIS结点出发的一条路径，仅当路径的终点为non-MIS结点时为valid swap。

## LSTwo

### Limited Search Depth

上述搜索算法的最坏时间复杂度为$O(d^V)$，为指数级别。

LSTwo限制其搜索深度为2，理由是深度为3的搜索对结果的提升不大，该结论可以通过计算各种情况的概率并且排除无效搜索得到，详见原论文中的说明。

LSTwo算法只需要枚举2 hop内邻居的情况，即可判断出是否存在valid swap。时间复杂度为$O(d^2)$.

### Approximation Error

由于LSTwo产生了近似解，尽管易知每一次update的近似误差不超过1，但关键在于k步近似的累计误差。

文章由归纳法证明，$k$步近似的累计误差同样不超过$k$。

假设第$k$步的累计误差不超过$k$，则由于第$k+1$步更新之后精确解的变化不超过1，近似解的变化也不超过1，且尽管这两个变化都不确定，但其变化方向都是确定的（根据加边、减边等四种情况只能是增加或减小中的一种）。因此第$k+1$步的累计误差也不会超过$k+1$.



## LazySearch

LazySearch基于对于Search Tree的观察，其中的某个结点，甚至于某个子图可能会被访问多次，但该访问通常是无效的。对于一个unbesieged结点，其被访问多次的代价很大，但却没有任何收获。LazySearch将每一次搜索中没有得到valid swap的non-MIS结点标记未invalid避免后续的搜索。这样做的理由有：

* 该结点很可能真的为unbesieged结点
* 该结点不是unbesieged结点，但由于MIS通常有多个解，放弃该结点并不会影响最终结果

这样做尽管存在代价，但其好处很多：

* 避免了多次访问unbesieged结点
* 对于non-MIS结点，每次对其访问都必须访问其所有的MIS邻居结点，该访问的代价很高

---

由于对于一次搜索，每条边至多被访问一次，若是失败的搜索则由于LazySearch的优化该边不会再被访问，若是成功的搜索则找到了答案，因此算法的复杂度为$O(E)$,为线性时间的算法。

## Besieged-Based Prunning

根据之前的分析，判断valid swap的充要条件是结点的besieged状态。

而某些结点的besieged状态可以被提前判断：当该结点的某些邻居结点在该结点之前访问时，该结点的besieged状态就可以被确定。这种根据访问顺序决定besieged状态的判断称为conditional besieged state。

对于小度数结点（degree-one or degree-two），conditional besieged state的判断是简单的。其类似于

文章 [Computing A Near-Maximum Independent Set in LinearTime by Reducing-Peeling](https://dl.acm.org/doi/abs/10.1145/3035918.3035939) 中的low-degree reduction在Dynamic MIS中的拓展。



### Degree-one Besieged Prunning

对于度数为1的non-MIS结点，如果其邻居在其之前访问，则其必为swap search中的叶子结点，则该结点为besieged结点。

对于度数为1的MIS结点，如果其邻居在其之前访问，则其必为swap search中的叶子结点，则该结点为unbesieged结点。



### Degree-two Besieged Prunning

仅考虑Triangle Case，设度数为2的MIS结点为$u$，其两个邻居为$x,y$，且$(x,y) \in E$ ，此时三个结点构成三角形。

若$x,y$的任意一个结点在$u$之前被访问，由于$x,y$为连边的non-MIS结点，则$u$一定搜为索的尽头，也即为叶子结点。

此时可以确定$u$的状态为unbesieged若其任意邻居在其之前被访问。



### Computing Besieged Graph

对于Besieged Prunning，不仅只能用在degree-one or degree-two结点上，重要的是该操作可以递归进行。

令$u \leftarrow v$，表示若$u$在$v$之前访问，则$u,v$的besieged状态都被确定。上面的degree-one or degree-two besieged prunning所针对的结点$v$都是low-degree 的边缘结点，下面考虑将其推广到其他结点，如结点$u$及其邻居结点。

考虑$u$的状态未被确定的情况，也即$v$在$u$之前访问的情况，此时可以删除已经被访问的结点$v$，删除后可能造成结点$u$的度数变为1或2，则可以递归地确定其他结点的conditional besieged。

由此，将构成一个besieged graph，仅当沿着图上的边的顺序访问时，结点的besieged状态是需要继续搜索的，否则其状态可以直接得出。根据besieged的传递和unbesieged的传递，共将形成两张图，分别用于判断两种关系。

考虑besieged graph的构建，由于每次构建都会删除图中的一条边，其复杂度为$O(E)$.



### Updating Besieged Graph

由于Besieged Graph为局部传递关系，因此在动态图更新中，一次更新仅会影响Besieged Graph的后代结点，而其前驱结点由于其局部关系并未发生变化，则之前的Besieged Graph的关系仍然成立。

因此，仅需要考虑Vertex Deletion，Edge Deletion和Edge Addition对Besieged Graph的不同影响，当可能造成影响时需要重新计算一次从每个端点开始的Besieged Graph，三种情况的详细分析详见论文。

此处以Edge Addition的情况简略说明要点，若新增边$(u,v)$，则如果新增便存在一个端点原先在Besieged Graph当中，则该Besieged Graph可能可以沿着该边传递，需要调用算法进行计算。



## Petal-Based Prunning



### Defintion of k-Petal 

k-petal定义为移除图中的某些结点后，剩余所有结点的度数不超过k的最大结点集合。

k-petal问题可以看作是MIS问题的推广，考虑0-petal问题，也即寻找一个最大集合，使得该集合不连通（移除其他结点后集合内结点度数均为0）。因为MIS问题是NP难问题，k-petal问题也是NP难问题。

---

引入k-petal问题的关键是，由于k-petal问题是MIS问题的推广，且两个问题有很多内在相似性和一致性。一定不在k-petal中的结点，通常也不在MIS中，因此这些结点可以被预先Pruning，此即Petal-Based Prunning。

### Effective Computation

首先，度数小于k的结点一定为k-petal中的结点。（用反证法与k-petal的最大性质可得）

因此，可以在$O(V)$时间内得到一定在k-petal中的结点集合。

再者，如果一个结点与k-petal中的结点连边数超过$k$，则根据k-petal的定义其一定不在k-petal中。因此，可以移除度数超过$k$的结点，若按度数降序排列，该操作的复杂度为$O(V \log V)$.

由此，可以得到不在k-petal中的结点，将其剪枝，认为其也不大可能在MIS中出现。



