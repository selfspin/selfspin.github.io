---
title: '随机变量的收敛'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计
---



随机变量的收敛性，包括几乎处处收敛、依概率收敛、依分布收敛、依范数收敛，以及其相互之间的蕴含关系证明。

<!--more-->



## Definition

* 依概率收敛($p$)，$\lim P(X_n = X) = 1$ 
* 几乎处处收敛($a.s.$)，$P(\lim X_n = X ) = 1$
* 依分布收敛($d$)，$\lim F_n(x) = F(x)$
* 依范数收敛($L_p$)，$\lim E[X-X_n]^p = 0$



## Relationship

在上述收敛性中，最弱的是依分布收敛，其次是依概率收敛，而较强的是依范数收敛和几乎处处收敛。

可以证明其有如下的蕴含性

### $a.s \rightarrow p$

几乎处处收敛本质上为给定了一个数列后的数列收敛性。

用分析的语言表述为，$\forall \epsilon>0, \exists n, \forall k >n, \vert X_k- X \vert < \epsilon$，该事件发生的概率为1.

令$n \rightarrow \infty$, $\forall \epsilon > 0, P(\cap_{k=n}^{\infty} \vert X_k -X \vert < \epsilon) = 1$

又$1 = \lim P(\cap_{k=n}^{\infty} \vert X_k -X \vert < \epsilon) \le \lim P(\vert X_n - X \vert < \epsilon)$

因此，有$\lim P(X = X_n) = 1$ 

也即，几乎处处收敛蕴含了依概率收敛。

利用上述的证明过程，也可以给出几乎处处收敛的一个充要条件: $ \sum_{n=1}^{\infty} P(\vert X - X_n \vert > \epsilon) < \infty $

### $L_p \rightarrow p$

证明用到切比雪夫不等式即可。

$ \lim P(\vert X - X_n \vert < \epsilon) \le \lim \frac{E \vert X- X_n \vert^p}{\epsilon^p}=0$

也即，依范数收敛蕴含了依概率收敛。常用的范数取2范数，此时也称作均方收敛。

### $ p \rightarrow d$

证明用到了分布函数与概率之间的关系。

$F_n(x) = P(X_n \le x) = P(X_n \le x, X \le x+\epsilon)+P(X_n \le x, X > x+\epsilon) \le F(x+\epsilon) + P(\vert X - X_n \vert > \epsilon)$

类似地，$F(x - \epsilon) \le F_n(x) + P(\vert X -X_n \vert > \epsilon)$

合并两式可得，$F(x-\epsilon) - P(\vert X - X_n \vert > \epsilon) \le F_n(x) \le F(x+\epsilon) +P(\vert X - X_n \vert > \epsilon)$

利用分布函数$F(x)$的连续性以及极限的夹逼定理可以得到，$F_n(x) \rightarrow F(x)$.

也即，依概率收敛蕴含了依分布收敛。



###  $d \rightarrow p, P(X = c)=1$

当$X$的分布是一个常数的时候，依分布收敛可以推出依概率收敛。

当$X$的分布为常数的时候，$F_n(x) \rightarrow F(x) =  I[x \ge c]$.  

$P(\vert X_n -c\vert > \epsilon ) \le F_n(c- \epsilon) + 1 - F_n(c + \epsilon) = 0$





