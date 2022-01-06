---
title: '一致性，随机树，随机森林'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---



本文讨论基于划分机制的机器学习算法的贝叶斯一致性，主要以随机决策树和随机森林为例。



<!--more-->



在 [代理性，最优性，可学性，一致性](https://truenobility303.github.io/Bayes-Consistency/) 中我们讨论了代理损失函数的贝叶斯一致性，此处讨论另一大类机制划分机制的一致性。

## Bayes Consistency of Partition Method

划分机制是一大类常用的机器学习模型的框架，其思想是将样本空间划分为不相交的部分，在每个部分是基于多数投票法得到预测的结果，基于划分机制的模型有K近邻、决策树、随机森林等。

贝叶斯一致性研究基于训练集学习得到的模型，是否可以随着样本容量的增加而趋近于贝叶斯最优分类器。

首先我们证明对于划分机制得到的机器学习模型，如果假设我们希望模型学得的条件概率为 $\eta(x)$ 在样本空间上连续，只要满足划分足够密集，且每个区域 $\Omega(x) $ 内投票的样本数 $N(x)$ 足够多，此时满足贝叶斯一致性。

基于每个区域 $\Omega(x)$ 可以得到基于多数投票法的预测为 $\hat\eta(x) $, 其首先应该趋近于该区域的条件概率 $\bar \eta(x) = E[ \eta(x), x \in \Omega(x)]$ , 进而当区域被切分得足够细致得时候，可以趋近于真实的概率 $\eta(x)$ ,从而达到贝叶斯一致性的效果，也即此时模型得到的分类器 $f$ 将趋近于贝叶斯最优分类器 $f_{\star}$ 的表现，将定理叙述如下，


$$
\begin{align}
\text{If } \text{diam}(\Omega(x)) &\rightarrow 0, N(x) \rightarrow \infty \text{With Prob. 1}\\
\text{Then } R(f) &\rightarrow R(f_{\star}) \text{ With Prob. 1}
\end{align}
$$


其中损失函数 $R$ 为0-1损失函数的期望损失，下面我们正式证明该定理，


$$
\begin{align}
\forall \epsilon,R(f) - R(f_{\star}) &= E_{\text{sign}(\hat \eta(x)) \ne \text{sign} (\eta(x))}[\vert 2 \eta(x) - 1 \vert ] ,\text{By A Simple Calculation} \\
&\le 2 E_{\text{sign}(\hat \eta(x)) \ne \text{sign} (\eta(x))} [\vert \eta(x) - \hat \eta(x) \vert \\
&\le 2E_x[\vert \eta(x) - \hat \eta(x)] \\
&\le 2E_x[ \vert \hat \eta(x) - \bar \eta(x) \vert] + 2 E_x[\vert \bar \eta(x) - \eta(x)\vert ] \\
&\le 2E_x[ \vert \hat \eta(x) - \bar \eta(x) \vert] + \frac{\epsilon}{2} , \text{By Continuous of } \eta(x) \text{And } \text{diam}(\Omega(x)) \rightarrow 0 \\
&\le 2E_{x,N(x)} [ \vert \hat \eta(x) - \bar \eta(x)\vert] + \frac{\epsilon}{2} \\
&=  2E_{x,N(x)} [ \vert \sum_{x_i \in \Omega(x)} \frac{I(y_i=1)- \bar \eta(x) \vert}{N(x)} ] + \frac{\epsilon}{2} \\
&\le 2E_x\sqrt {E[\vert \sum_{x_i \in \Omega(x)} \frac{I(y_i=1)- \bar \eta(x)}{N(x)} ]^2} + \frac{\epsilon}{2}, \text{By Jessen's Inequality} \\
&=2E_x\sqrt {\frac{1}{N(x)^2}E[\sum_{x_i \in \Omega(x)}I(y_i=1)- \bar \eta(x) ]^2} + \frac{\epsilon}{2} \\
&= 2E_x\sqrt{\frac{Var(X)}{N(x)}} + \frac{\epsilon}{2}, \text{Let } X \text{ Be } I(y_i = 1) \\
&=2E_x\sqrt{ \frac{\bar \eta(x) (1-\bar \eta(x))}{N(x)}} + \frac{\epsilon}{2} \\
&\le E_x\sqrt{ \frac{1}{N(x)}} + \frac{\epsilon}{2} ,\text{By } \bar \eta(x) (1-\bar \eta(x)) \le \frac{1}{4}\\
&=E_{x,N(x) \ge N} \sqrt{ \frac{1}{N(x)} } +\frac{\epsilon}{2} ,\forall N ,\text{By } N(x) \rightarrow \infty \\
&\le \frac{1}{\sqrt N} + \frac{\epsilon}{2} \\
&\le \epsilon ,\text{Let } N \ge (\frac{\epsilon}{2})^2\\
\end{align}
$$


该定理给出了划分机制贝叶斯一致性的充分条件，将用来证明各种基于划分机制的机器学习模型的贝叶斯一致性。



## Bayes Consistency of Random Decision Tree

考虑随机决策树的机器学习模型，考虑 $d$ 维的样本空间，也即自变量有 $d$ 个属性，每个属性取值范围为 $[0,1]$. 

算法迭代 $k$ 次， 每一次迭代过程中随机选取区域，每次划分随机选择一个属性，之后随机选择$[0,1]$中的一个数对该属性进行划分。

我们下面将证明随机决策树算法生成的区域将足够细致，而只需要样本容量 $m$ 相对于迭代次数 $k$ 为无穷大量，则落入每个区域的样本将足够多，从而使得上述普遍定理的条件得到满足，从而证明随机决策树的贝叶斯一致性。



首先证明则落入每个区域的样本将足够多， 以概率1成立，只需证明对于任意给定的数 $N$, 区域内样本数小于 $N$ 的概率为0，


$$
\begin{align}
P(N(x) \le N) &= E_{x} [ I(N(x) \le N)] \\
&= \sum_{i=1}^kE [ I(N(\Omega_i) \le N) I(x \in \Omega_i)] \\ 
&= k E_{N(\Omega_i) \le N,x}[I(x \in \Omega_i)] \\
&=k E_{N(\Omega_i) \le N}[\frac{N_i}{m}] \\
&\le k E_{N(\Omega_i) \le N}[\frac{N}{m}] \\
&\le \frac{kN}{m} \\
&\rightarrow 0 ,\text{Gievn } N \text{ And } \frac{k}{m} \rightarrow 0
\end{align}
$$


因此可以得到落入每个区域的样本将足够多，也即


$$
P(N(x) = \infty) \rightarrow 1
$$


再者我们证明区域的划分将足够细致，由于直接证明较为困难，

首先通过证明每个区域 $\Omega(x) $ 被划分的次数 $T(x)$ 将足够多，进而证明每个区域的边长 $L(x)$ 将足够小，


$$
\begin{align}
E[T] &= \sum_{i=1}^k \frac{1}{i} \\
&= \sum_{i=1}^k \log (\frac{1}{i}+1) ,\text{By } x \ge \log(x+1)\\
&\ge \sum_{i=1}^k \log \frac{i+1}{i} \\
&= \log (k+1) \\
&\ge \log k \\
Var[T] &= \sum_{i=2}^k \frac{1}{i}(1-\frac{1}{i} ) \\
&\le  \sum_{i=2}^k\log \frac{1}{1-\frac{1}{i}(1-\frac{1}{i})} ,\text{By } x \le \log \frac{1}{1-x}\\
&= \sum_{i=2}^k \log \frac{i^2}{i^2 - i+1}\\
&\le \sum_{i=2}^k \log \frac{i^2}{i^2 - i}\\ 
&= \sum_{i=2}^k \log \frac{i}{i - 1}\\
&= \log k \\
P(\vert T - ET \vert \ge \frac{ET}{2}) &\le \frac{4 Var[T]}{E[T]^2} ,\text{By Chebyshev Inequality} \\
&\le \frac{4 \log k}{\log^2 k} \\
&= \frac{4}{ \log k} \\
P(T \ge \frac{\log k}{2}) &\ge P(T \ge \frac{ET}{2}) \\
&\ge 1-\frac{4}{ \log k }\\
&\rightarrow 1, \text{With } k \rightarrow \infty
\end{align}
$$


总结经过推导得到了每个区域都将被无限次划分，随着迭代次数趋近于无穷，可以得到，


$$
P( T = \infty) \rightarrow 1
$$
进而可以计算对每一个属性的划分次数 $S$，从而得到每个属性的边长 $L$ 的期望，给定该区域被划分的总次数 $T$ 可以计算出该属性被划分的次数 $S$, 给定$S$ 每次划分一次都会对属性的边长 $L$ 有期望意义下的衰减，


$$
\begin{align}
E[L] &\le E_{T,S}[\prod_{i=1}^S \max (U_i, 1-U_i)] \\
&=E_{T,S}[\prod_{i=1}^S 2 \int_{0}^{\frac{1}{2}} U_i dU_i] \\
&=E_{T,S}[\prod_{i=1}^S \frac{3}{4}] \\
&=E_{T,S}[(\frac{3}{4})^S] \\
&=E_T[\sum_{S=1}^T\binom{T}{S} (\frac{1}{d})^S (1-\frac{1}{d})^{T-S} (\frac{3}{4})^S] \\
&=E_T[\sum_{S=1}^T\binom{T}{S} (\frac{3}{4d})^S (1-\frac{1}{d})^{T-S}] \\
&=E_T [(1+\frac{3}{4d} -\frac{1}{d} )^T] \\
&=E_T[(1-\frac{1}{4d})^T] \\
&\rightarrow 0, \text{With } T \rightarrow \infty
\end{align}
$$



最后对于 $d$ 维空间，期望直径和期望变成存在关系，可以得到期望意义下区域的直径也趋于0，


$$
\begin{align}
E[\text{diam}^2(\Omega(x))] &= E[\sum_{i=1}^d L_i^2]\\
&= \sum_{i=1}^d E[L_i^2] \\
&\le \sum_{i=1}^d E[L_i] ,\text{By } L_i \le 1 \\
&\rightarrow 0 \\
\forall D, P(\text{diam}(\Omega(x)) \ge D) &= P(\text{diam}^2(\Omega(x)) \ge D^2) \\
&\le \frac{E[\text{diam}^2(\Omega(x))]}{D^2} \\
&\rightarrow 0  \\
P(\text{diam}(\Omega(x) = 0) & \rightarrow 1
\end{align}
$$


最终我们证明了随机树的构造满足对于划分机制的贝叶斯一致性的充分条件，因此随机树满足贝叶斯一致性。



## Bayes Consistency of Random Forest



随机森林基于随机决策树生成，假定生成 $R$ 棵独立的随机决策树 $T$，则随机森林的预测结果为 $R$ 可随机决策树 $\bar T$ 的多数表决结果。

我们已经知道，每一棵随机决策树都将趋近于贝叶斯最优分类器，直观上我们可以容易理解其多数表决的结果也将趋近于贝叶斯最优分类器，下面我们将其形式化表达，


$$
\begin{align}
R(\bar T) - R({T_\star}) &= E_{\text{sign}(\bar T) \ne \text{sign}(T_{\star})} [\vert 2 \eta - 1 \vert],\forall x, \eta = \eta(x)\\
&=E_x[\vert2\eta -1 \vert E[I({\text{sign}(\bar T) \ne \text{sign}(T_{\star}))}]] \\
&=E_x[\vert2\eta -1 \vert E[\sum_{i=1}^R I(\text{sign}(T_i) \ne \text{sign}(T_{\star})) \ge \frac{R}{2} ]] \\
&\le E_x[\vert2\eta -1 \vert  \frac{2 E[\sum_{i=1}^R I(\text{sign}(T_i) \ne \text{sign}(T_{\star})] }{R}] ，\text{By Markov Inequality}\\
&=2 E_x[\vert 2 \eta - 1 \vert E[I(\text{sign}(T) \ne \text{sign}(T_{\star})]] \\
&=2E_{\text{sign}(T) \ne \text{sign}(T_{\star})}[\vert 2 \eta -1 \vert] \\
&=2 (R(T) - R(T_{\star})) \\
&\rightarrow 0
\end{align}
$$


证明的关键在于使用每一棵随机决策树的误差控制整个随机森林的误差。

最终我们成功地证明了基于上述构造的随机森林同样具有贝叶斯一致性。

