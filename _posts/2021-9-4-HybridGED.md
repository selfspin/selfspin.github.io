---
title: 'Hybrid Lower Bound for Graph Edit Distance'
toc: true
excerpt_separator: <!--more-->
tags:
  - 图算法
  - 图编辑距离

---



论文阅读笔记：[Efficient Graph Similarity Search Over Large Graph Databases](https://ieeexplore.ieee.org/abstract/document/6880803)

<!--more-->

## Abstract

文章提出了针对图编辑距离（Graph Edit Distance，GED）两种不同的lower bound体系，并且基于两者设计了一种比两者都更紧的Hybrid Lower Bound。并且针对实际应用中多个图的编辑距离计算问题，设计了称为u-tree的数据结构用于剪枝，加速实际场景下的图查询任务。

## Graph Edit Distance

图编辑距离指的是从一张图编辑为另一张图的最小代价，其中可选用的编辑操作有加点、减点、加边、减边、改变点标签、改变边标签，每步代价都为1. 是一个经典的解决图近似搜索的NP难问题，且应用广泛。

## Branch-Based Lower Bound

基于图的结构特点，可以给出一个GED的上界。首先将图分解为一些结构，GED可以认为是图A的每个结点到图B的每个结构之间的一个匹配，也即一个二分图匹配问题，求其最小带权匹配，可以得到GED的上界。之前的经典方法，基于将图分解为star结构，但其缺点是，给出的GED上界和图的结点的度数有关，当图的最大结点的度数很大时，给出的上界较为宽松。

而本文章提出的基于branch的上界，和结点的度数无关。

branch定义为一个结点以及从这个结点出发的所有边，可以用中心点$V$的标签，和其连边的边集$E$的标签表示。

定义两个不同branch之间的编辑距离为:



$$
d(A,B) = I(V_A=V_B) + \frac{\max(\vert E_A \vert,\vert E_B \vert)-\vert E_A \cap E_B \vert }{2}
$$



采用上述定义的branch的编辑距离作为二分图中的边权，求解该二分图的最小带权匹配，可以证明该匹配的权重是GED的下界。

证明：当GED为0时，最小带权匹配也为0，考虑每一步编辑，分为两种:

* 对结点的编辑，仅仅影响一个branch，仅改变1的权重
* 对边的编辑，仅仅影响两个branch，除以2后也仅改变1的权重

也即，每一步编辑至多带来1的权重代价，则最终的总权重为GED的下界。

---

上述算法比其基于star structure的算法的本质改进在于更改为branch之后，编辑所影响的branch个数变小了，不像编辑影响的star与顶点度数有关。上述算法求解二分图最小带权匹配的复杂度为$O(V^3)$，但定义如下的branch编辑距离，可以在更小的复杂度的情况下得到一个较松的下界，证明类似，此处暂略：



$$
d(A,B) = I(V_A=V_B) + \frac{I(E_A=E_B)}{2}
$$



因为branch可以用中心点$V$的标签，和其连边的边集$E$的标签表示，因此可以将所有的branch排序后，在$O(V \log V)$的时间内求解这个特殊的带权二分图最小匹配问题。



## Partition-Based Lower Bound

基于图分解给出另一种下界。

将图A分解为不相交的几个部分，则这些部分中在图B中找不到子图匹配的个数也为一个下界。

证明：GED为0时每个部分都属于图B的子图，而每一步编辑最多仅能改变一个部分，因此将图A编辑为图B至少需要这么多步。

---

下面的关键在于寻找一个分解，首先将该问题转化为较简单的存在性判断问题：给定一个阈值$t$,判断是否存在一种分解方式，使得该分解方式给出一个至少为$t$的阈值，而该问题是NP难的：

考虑$t=1$的情况，也即图A不分解，转化为NP难问题：子图同构问题。

对于该NP难问题，给出一种较简单的算法，根据Partition子部分的大小分类讨论：

* 对于大小（边数+点数）不超过3的部分：单个结点、一条边及其两个端点等，寻找这部分的子图同构可以在$O(V+E)$时间内完成
* 其他部分的子图同构判断

采用上述方法存在几个提前终止策略：

* 先对较小的部分进行判断，并且如果已经达到了$t$，则可以提前终止
* 若第一部分找到的界为$x$，则对于其他大小均大于3的部分，若$\frac{V+E}{3} < t-x$,则不可能找到，也可以提前终止
* 限制第二部分的大小，当大小超过$T$时停止不再计算



## Hybrid and Tighter Lower Bound

Hybrid Bound基于通配符的思想，首先利用Partion得到了一系列子部分以及一个Partition-Based Lower Bound $d_P$, 然后在所有未匹配子部分枚举一步通配编辑（将一个结点/边更改为一个可以匹配所有点/边的通配符），计算产生的带通配符图到另一个图的Branch-Based Lower Bound $\lambda_B$, 定义Hybrid Lower Bound：



$$
d_H=d_P+\lambda_B
$$



下面证明这是一个比上述两种界都更紧的下界：

首先，由于$d_B \ge 0$,因此$d_H \ge d_P$,因为Hybird Lower Bound比Partition-Based Bound更紧

再者，记带通配的图中取到$d_B$的为图$C$，考虑将图A先编辑为图C再编辑为图B，根据Branch-Based Lower Bound的性质，$\lambda_B \le GED(B,C) $, 因此$GED(A,B)=GED(A,C)+GED(C,B) \ge d_P+ \lambda_B$,因此$d_H$确实是一个下界。

再证明，$d_H$比Branch-Based Lower Bound给出的界$\lambda'=\lambda(A,B)$更紧，因为$\lambda'=\lambda(A,B)=\lambda(A,C)+\lambda(A,B) \le d_p + \lambda_B$

因此，Hybrid Lower Bound给出一个下界，且该下界更紧，本质原因是通配符编辑一定是满足条件的编辑，Hybrid Lower Bound解决了两种方法的弊端：

* Partition-Based Lower Bound仅考虑每个子部分的一步编辑，但却没有考虑后续的编辑
* Branch-Based Lower Bound可以考虑多步编辑，但却应用了二分图匹配计算了多步编辑的宽松的下界

而Hybrid Lower Bound将Partition-Based Lower Bound的第一步编辑用通配符编辑计算，而后续的其他编辑使用二分图匹配计算。



## U-Tree

U-Tree在于实际应用需求中需要同时求解一个查询图$Q$和数据库中多个图的GED。基于的思想是，将两个图的branch并起来构成一个更大的集合，构成U-Tree中的一个联合结点，该结点维护两个图的统计量信息：Partition-Based Lower Bound中大小不超过3的结构，Barnch-Based Lower Bound中branch的联合。 相较于两个图分别到查询图的编辑距离，不管是每个子部分的中可能存在子图同构的数目（Partition-Based Lower Bound），还是branch中可能找到的更优的匹配（Branch-Based Lower Bound），都可以得到改进，因此联合结点的GED也给出了一个下界。设查询操作中允许的GED阈值为$t$,当U-Tree中一个联合结点的GED已经超过了该阈值的时候，说明器所有子孙结点的GED必然也超过该阈值，因此剪枝成立。

显然，当U-Tree相邻两个子孙结点的各项统计量都接近的时候，其组成的联合结点可以给出更紧的上界，因此应依据该原则构建U-Tree.

