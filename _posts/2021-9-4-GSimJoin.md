---
title: 'GSimJoin：Path-Based Graph Edit Distance'
toc: true
excerpt_separator: <!--more-->
tags:
  - 图算法
  - 图编辑距离

---

论文阅读笔记：[ICDE'12 Efficient Graph Similarity Joins with Edit Distance Constraints](https://ieeexplore.ieee.org/abstract/document/6228137/)

<!--more-->

## Abstract

解决GED的查询问题：给定一张图，查询数据库中所有与查询图的GED小于阈值$\tau$的数据图。算法主要分为两个部分：

* Filter，过滤掉不可能作为候选的数据图，只要解决其中的过滤条件

* Estimate When Seaching，调用$A^{\star}$算法进行搜索，主要解决其中的估值函数

算法主要的思想基于path分解，也即将图分解为不同path组成的q-gram，利用q-gram进行过滤（Filter）和估值（Estimate）



## Path-Based Q-Gram

### Difference From Tree-Based

之前的方法采用的可以归为Tree-Based Method，采用的主要思想是将图的结构拆解为从一个结点出发的深度不超过$q$的BFS树作为Q-Gram。但Tree-Based Method中，一次图编辑操作所影响的最多Q-Gram数目为:



$$
N_{tree}= 1+\frac{d((d-1)^q-1)}{d-2}
$$



其中$d$为图中结点的最大度数，证明只需要假设BFS树中的所有结点度数都为$d$，并使用等比数列求和即可。

而文章改进的思想在于Path-Based，也即将图的结构拆解为一条路径，此时记包含某个结点的最大路径数目为$\gamma$,则易证得每一次图编辑操作所影响的Q-Gram数目最多为$\gamma$. 

证明也是显然的，只需考虑所有可能的图编辑操作，详见论文。

### Size Filter

比较两个图边集和点集的大小，若大小差大于阈值$\tau$，则不可能为满足条件的解。

上述Filter的成立是显然的



### Count Filter

将两个图$r,s$分解为相应的Q-Gram之后，类似于之前的方法，可以采用二分图匹配来估计得到GED的一个下界，但这样做的缺点是复杂度较大而且没有利用到阈值$\tau$的限制条件。基于Q-Gram的Filter只需要找到一个超过$\tau$的下界即可。

由此，可以提出Count Filter：根据不相同Q-Gram的数目（mismatching structures）进行Filter操作。

**记一次图编辑操作所能影响到的最大Q-Gram数目为$x$，而两个图不相同的的Q-Gram数目为$y$，则当$\tau x> y$的时候，GED超过阈值。**

由此，Path-Based Q-Gram最显著的优势在于每次图编辑操作所影响的数目更小，从而某种意义上Filter的效果会更好



## Exploiting Mismatching Structures

上述的Count Filter利用了mismatching structures的数目进行FIlter，但更进一步可以利用其结构信息和标签信息进行Filter。

### Label Filter

利用标签信息的思想是直观的：

**当图不匹配的标签数目（点标签或者边标签）超过阈值$\tau$的时候，也即当$\Gamma(L_V(r),L_V(s))+\Gamma(L_E(r),L_E(s)) > \tau$时，显然不可能称为候选解。**

其中$\Gamma(A,B)=\max(\vert A \vert,\vert B \vert)-\vert A \cap B \vert$ ,表示两个集合不相同的标签数目。



### Minimal Edit Filter

而对于所有的mismatching structures，可以寻找影响到所有mismatching structures所至少需要的图编辑数目，本质上为一个NP难问题：最小点覆盖问题。

因此我们有如下的过滤条件：

**当影响所有mismatching structures的最小操作数大于$\tau$的时候，也不可能为候选解**

对于上述最小图编辑（最小点覆盖）问题，可以使用精确解法或者使用近似解法。



### Edit Distance Estimation

上述的Label Filter和Structure Filter对于每一个mismatching structures组成的连通子图来说都是成立的，简单推广即可，证明暂略。

可以对每个连通分量同时根据Label Filter和Minimal Filter所得到的结果取较大的，则得到每个连通分量的图编辑距离的下界。

所有连通分量的下界和加起来，即为整张图的编辑距离的下界，可以作为一个估计值出现。



## Prefix Filter

在Count Filter中，需要比较Q-Gram中相等的个数，此时需要根据Q-Gram排序。根据启发式思想，可以根据逆词频排序，依据是较少出现的Q-Gram更具有区分度。

根据Count Filter的结论，在比较相等的Q-Gram时，依照顺序比较Q-Gram，当不匹配的Q-Gram数目达到$\tau x$的时候，可以排除该数据图作为查询结果。

考虑优化上述算法，本质上其实上述算法是在：

**寻找至少需要$\tau$次操作才能影响其中的所有Q-Gram的一个前缀，当该前缀中所有的Q-Gram都不匹配的时候，可以排除该数据图作为查询结果。**

上述前缀显然可以取$\tau x$，此时也即Count Filter，但上述前缀实际上可以更短，此时Filter的效果会更好。因此可以求解一个最短的前缀。

显然，这个最短的前缀长度满足单调性，可以用二分搜索解决，搜索范围为$[\tau,\tau x]$，二分判断一个前缀的最小图编辑问题。

在二分的时候，仍然可以利用最小点覆盖的近似解法，先得到最短前缀的上界，理由是近似解法求得的图编辑距离相较于最小图编辑距离较大，因此求得的为最小前缀长度的一个上界。



## Algorithm

对于所有的图，前缀是确定的，可以预先处理。

同时建立索引，对于一种Q-Gram，记录下前缀包含该Q-Gram的所有数据图，此时的数据图也即满足Prefix Filter条件的图。

完整算法流程如下：

* 对于查询$Q$,根据上述前缀索引，返回所有满足Prefix Filter条件的候选图
* 进行最简单的Filter判断，全局标签数目、集合数目的Filter: Global Label Filter and Size Filter，去除部分候选
* 比较共同的Q-Gram，进行Count Filter进一步筛选候选，同时返回mismatching structures
* 调用Edit Distance Estimation的方法，统计所有连通分量的图编辑距离之和的下界（估值），去除估值超过阈值的候选
* 对于上述所有Filter操作都未能去除的数据图，调用$A^{\star}$算法，其中每一步所需的估值函数仍然可以使用Edit Distance Estimation计算





