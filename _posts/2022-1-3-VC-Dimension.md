---
title: 'VC维，复杂度，泛化界'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---



使用增长函数和VC维衡量假设空间的复杂度，并且介绍常见分类器，如线性分类器、前馈神经网络、多数投票分类器的VC维，并且根据VC维以及概率不等式等给出关于有限维和无限维假设空间的泛化误差上界。



<!--more-->

内容参考了 《机器学习理论导引》.周志华等.  证明参考了计算理论相关的课程讲义。

未做说明的话，均认为本文考虑机器学习中常见的二分类问题。

## Complexity of Hypothesis



机器学习中，需要给定一个假设空间，利用某种评价准则（损失函数等）在假设空间中学习到一个最优的模型。

我们关注于假设空间的复杂度，在后面将看到该复杂度对于泛化误差界等也扮演着重要的作用。

对于给定的假设空间，可能是有限维的假设空间，此时可以直接用假设空间的维度，也即空间中假设的个数来直接衡量其复杂度。但对于无限维的假设空间，我们需要定义其对应的复杂度，也就是著名的增长函数和VC维。



### Growth Function

在引入VC维之前需要给出增长函数的定义，其定义为假设空间 $\mathcal{H}$ 对于大小为 $m$ 的样本集 $D$所能赋予的最大可能的标记数，在二分类问题上，该标记为 $+1,-1$ , 增长函数定义为：


$$
\begin{align}
\Pi(m) &= \max_{\vert D \vert = m } \vert \mathcal{H}_{D} \vert \\
&=\max_{\vert D \vert = m } [\vert (h(x_1), h(x_2), ... h(x_m)) \vert, h \in \mathcal{H}]
\end{align}
$$


### VC-Dimension

如果假设空间可以实现大小为 $m$ 的样本集的所有可能的标记结果，则称为打散（Shattering）该样本集。VC维定义为假设空间可以打散的样本集的最大大小，也即：



$$
VC(\mathcal{H}) = \max_m [\Pi(m) = 2^m]
$$


### Sauser's Lemma

Sauser给出了增长函数和VC维之间的关系，由于在很多情况下我们更加关注VC维或者VC维的上界，但VC维较难以计算。但增长函数的计算根据定义可能是简单的，因为该引理对于很多机器学习，可以计算出VC维的上界：


$$
\Pi(m) \le \sum_{i=0}^d \binom{m}{i}, \forall m ,d = VC(\mathcal{H})
$$


证明使用了数学归纳法，首先假设对于 $m-1$ 都成立上述的式子，考虑使得样本容量为 $m$ 的增长函数所对应的数据集 $D$, 该数据集对对应的所有可能的标记数目实际上可以被两个大小为 $m-1$ 的子数据集的标记数目表示。

已知 $\mathcal{H}_D = (h(x_1),h(x_2),...,h(x_m)) $,  对于$h(x_m)$ 仅有一种取值的情况，定义 $\mathcal{H}_{D_1} = (h(x_1),h(x_2),...,h(x_{m-1})) $.  

而对于$h(x_m)$ 有两种取值的情况，相应地定义 $\mathcal{H}_{D_2} = (h(x_1),h(x_2),...,h(x_{m-1})) $.

对于 $\mathcal{H}_{D_2}$ , 由于其满足样本集的最后一个维度被打散的限制，其VC维应该比原来更小，且最多为$d-1$， 据此可以递推得到，



$$
\begin{align}
\Pi(m) &= \vert \mathcal{H}_D \vert \\
&=\vert \mathcal{H}_{D_1} \vert + \vert \mathcal{H}_{D_2} \vert \\
&\le \sum_{i=0}^d \binom{m-1}{i} + \sum_{i=0}^{d-1} \binom{m-1}{i} \\
&=1 + \sum_{i=1}^d \binom{m-1}{i} + \sum_{i=1}^{d} \binom{m-1}{i-1} \\
&=1 + \sum_{i=1}^d (\binom{m-1}{i} +  \binom{m-1}{i-1}) \\
&= 1+ \sum_{i=1}^d \binom{m}{i} \\
&=\sum_{i=0}^d \binom{m}{i}
\end{align}
$$



更进一步，当 $m \ge d$， 也即样本容量大于VC维的时候，可以得到增长函数的上界，


$$
\begin{align}
\Pi(m) &\le \sum_{i=0}^d \binom{m}{i} \\
&\le  (\frac{m}{d})^d \sum_{i=0}^d \binom{m}{i} (\frac{d}{m})^i \\
&\le (\frac{m}{d})^d \sum_{i=0}^m \binom{m}{i} (\frac{d}{m})^i \\
&= (\frac{m}{d})^d (1+ \frac{d}{m})^m \\
&\le (\frac{m}{d})^d e^d \\
&= (\frac{em}{d})^d \\
&\le (em)^d
\end{align}
$$


## VC-Dimension of Classical Classifiers

本节给出一些经典的机器学习模型的VC维，据此更加具体地理解VC维的定义



### Linear Classifier

线性分类器是机器学习中常用地一类分类器，包括支持向量机（SVM，Support Vector Machine），对率回归（Logistic Regression）等模型，其模型的统一表达式为，


$$
h(x) = \text{sign}(w^T x+ b )
$$


本质上求解线性分类器的VC维相当于求解 $d$ 维空间的线性超平面的VC维，可以根据VC维的定义得到，



一方面，存在一个样本容量为 $d+1$ 的数据集 $D$ 使得其可以被线性超平面所打散，


$$
\begin{align}
\text{Given } x_0 &= 0,x_1 = e_1,x_2 = e_2,...x_d = e_d \\
\forall y_0,y_1,...y_d, \text{Let } w &= [y_1,y_2,...,y_d]^T, b = y_0\\
\text{Then } h(x_i)  &= \text{sign}(w^T x_i +b) = y_i 
\end{align}
$$


另一方面，任意样本容量为 $d+2$ 的数据集 $D$, 就算去掉其中的零向量，至少仍然会剩下 $d$ 维空间内的 $d+1$  个线性相关的向量，下面证明对于任意的 $x_0,x_1,.. x_{d+1}$ ， 其一定不可能被线性超平面所打散，不妨假设 $x_0 = 0$, 接着使用反证法即可， 


$$
\begin{align}
x_1 &= \sum_{i=2}^{d+1} a_i x_i ,\exists a_i \ne  0,  \\
w^T x_1 +b &= \sum_{i=2}^{d+1} a_i (w^Tx_i+b) \\
\text{Let } y_1 &= -1, y_{i} = \text{sign}(a_i), \forall 2 \le i \le d+1  \\ 
\text{sign} h(x_1) &=  \text{sign} (w^Tx_1 + b ) = \text{sign} ( \sum_{i=2}^{d+1} a_i (w^Tx_i+b)) = 1 \ne -1
\end{align}
$$




因此，我们证明了存在一个样本容量为 $d+1$ 的数据集 $D$ 使得其可以被线性超平面所打散，并且任意样本容量为 $d+2$ 的数据集 $D$, 其一定不可能被线性超平面所打散，也即线性分类器的VC维为 $d+1$. 



### Neural Networks

神经网络是强大的分类器，且在实践中被证明具有强大的拟合能力，但有一些观点认为神经网络很有可能仅仅是在过拟合某些数据集。本节我们希望衡量神经网络的VC维，根据著名的通用近似定理，使用Sigmoid激活的前馈神经网络可以以给定精度拟合任意函数，因此对于任意给定的数据集，神经网络都可以将其打散，其VC维应当为 $\infty$. 本节我们关心的是其VC维关于参数量的关系。 

该部分证明参考了 [CMSC 35900 (Spring 2008) Learning Theory](https://home.ttic.edu/~tewari/lectures/lecture12.pdf)



关于神经网络的通用近似定理，可以参见 [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap4.html) 中给出的证明，实际上使用到了Sigmoid函数可以逼近Sign函数，而使用Sign函数可以构造出分段的Step Function，从而达到以给定精度拟合任意函数的效果。因为本节我们仅仅考虑使用Sign函数激活的前馈神经网络，虽然失去了一般性，但是实际上相当于对于拟合能力最强的神经网络的VC维进行了分析。



使用Sign函数激活的前馈神经网络 $\mathcal{F }$ 实际上是一个复合函数，假设其一共有 $h$ 层，每一层有 $n_i$ 个神经元，每一个神经元都是形如 $\text{sign}(w^T x+b)$ 的二分类单元。也即最终的前馈神经网络相当于 $h$ 个函数的复合，而每一个一层函数 $h_i$ 的值域相当于 $n_i$ 个神经元的笛卡尔积，从而我们将前馈神经网络分解为每个神经元的笛卡尔积和复合函数的组合，而增长函数关于笛卡尔积和复合函数这两个运算可以根据定义得到其满足单调性，具体详见以下的证明，其中 $d_{ij}$ 表示第 $i$ 层的第 $j$ 个神经元的参数个数，而下式中的 $N$ 表示整个神经网络的参数总个数。令  $m$ 为该神经网络所能够打散的最大样本容量，则成立，


$$
\begin{align}
VC(\mathcal{F}) &= 2^m \\
&= \Pi_{f_1 \circ f_2 \circ ... \circ f_h}(m) \\
&\le \prod_{i=1}^h \Pi_{f_i}(m) \\
&= \prod_{i=1}^h \Pi_{g_{i,1} \times g_{i,2} \times ... \times g_{i,n_i}}(m) \\
&\le \prod_{i=1}^h \prod_{j=1}^{n_i} \Pi_{g_{ij}}(m) ,\text{With } g_{ij}(x) = \text{sign}(w_{ij}^Tx + b_{ij})\\
&\le \prod_{ij} (em)^{d_{ij}} ,\text{With } d_{ij} = VC(g_{ij}) \\
&= (em)^N , \text{Let } N = \sum_{ij} d_{ij}
\end{align}
$$


可以得到关于其VC维 $m$ 的不等式，


$$
\begin{align}
2^m &\le (em)^N \\
m &\le N \log em \\
em & \le eN \log em \\
\frac{em}{eN \log em} &\le 1 \\
\end{align}
$$


利用反证法，根据对复杂度上界的分析可以得到，


$$
\begin{align}
em &= O(N \log N) \\ 
\text{Otherwise, } em &\gt K N \log N , \forall K , N \rightarrow \infty\\
\text{Then, } 1 &\ge \frac{em}{eN \log em}  \\
& \gt \frac{K N \log N}{ eN \log K + eN \log N+ e N \log \log N } \\
&\gt \frac{K N \log N}{ 2eN \log N} \\ 
&= \frac{K}{2e} > 1 , \text{Let } K > 2e
\end{align}
$$


因此可以得到该前馈神经网络的VC维上界为 $O(N \log N)$ 



### Ensemble with Votings



类似的手段可以应用于集成学习模型中，对于机器学习中的集成学习模型，感兴趣的读者可以参见 [集成学习](https://truenobility303.github.io/Ensemble/) 

本节简单分析最简单的集成学习模型，采用多数投票法（Majority Vote  Point Classifiers）得到的模型，也即


$$
F(x) = \text{sign} (\sum_{i=1}^h f_i(x))
$$
假设每个子模型的VC维均为 $d$ ,  同样地令 $m$ 为该集成模型所能够打散的最大样本容量，则成立，


$$
\begin{align}
2^m &= \Pi_F(m) \\
&= \Pi_{g(x)}(m) , \text{With } g(x_1,x_2,..x_h) = \text{sign}(\sum_{i=1}^h x_i) \\
&\le \prod_{i=1}^h \Pi_{f_i}(m) \\
&\le (em)^{dh} 
\end{align}
$$


得到了类似的不等式，类似地可以得到多数投票法的VC维上界满足，


$$
VC(F)  = O(dh \log dh)
$$


## Generalization Bound

本节研究泛化误差上界与假设空间复杂度的关系。



### Finite  Hypothesis Space

首先研究有限假设空间的泛化误差界，分为可分和不可分两种不同的情况讨论，考虑使用0-1损失。

我们关心在训练集上得到的训练误差，或者称为经验损失 $\hat R$，与实际上总体数据集期望情况下的误差，或者称为期望损失 $R$ 之间的关系。设假设空间的维度为 $d$,  训练集的大小为 $m$. 

可分指的是在假设空间中存在一个假设可以将数据集正确分类，此时0-1损失为0.

而不可分指的是假设空间中不存在一个假设可以将数据集正确分类，此时0-1损失不可能达到0.

---

对于可分的情况，由于我们已知最优的假设一定满足在所有数据上的损失为0，因此其一定满足经验损失为0. 算法枚举假设空间中所有的假设，并且输出所有经验损失为0的假设，我们希望知道得到的这样的假设的泛化误差界，


$$
\begin{align}
P(R(h)> \epsilon , \hat R(h) = 0 ,\exists h \in \mathcal{H}) &\le d P(R(h) > \epsilon, \hat R(h) = 0) \\
&\le d (1-\epsilon)^m \\
&\le d \exp(-\epsilon m ) \\
&= \delta
\end{align}
$$
据此可以得到有限维假设空间可分情况的泛化误差界，对于假设空间中满足经验损失为 $0$ 其一定满足，


$$
\begin{align}
P(R(h) \le  \hat R(h) +\frac{1}{m} \log \frac{d}{\delta}) \ge  1- \delta ,\text{With } \hat R(h) &= 0 ,\forall h \in \mathcal{H} ,\\
\end{align}
$$


该泛化误差界也符合直观理解，其反映了当假设空间维度 $d$ 越大，模型学习的难度越大。而样本容量 $m$ 越大，泛化误差的上界越小，且泛化误差关于样本容量的收敛率为 $O( \frac{1}{m})$ .

---

下面我们考虑有限维假设空间但是不可分的情况，此时需要用到Hoeffding不等式，关于其证明可以参见 [概率不等式](https://truenobility303.github.io/Probabilistic-Inequality/) 


$$
\begin{align}
P(R(h) > \hat R(h) + \epsilon, \exist h \in \mathcal{H}) &\le d P(R(h) > \hat R(h) + \epsilon) \\
&\le d \exp(-2m \epsilon^2) ,\text{By Hoeffding Inequality}\\
&= \delta 
\end{align}
$$
据此可以得到，


$$
\begin{align}
P(R(h) \le  \hat R(h) + \sqrt{\frac{1}{2m} \log \frac{d}{\delta}}) \ge  1- \delta ,\forall h \in \mathcal{H} \\
\end{align}
$$


对于不可分问题，利用Hoeffding不等式得到了类似的结果，但可以发现此时模型学习变得更为困难，

此时收敛率由相较于原来的 $O(\frac{1}{m})$ 变为 $O(\frac{1}{\sqrt m})$ ， 也即需要更多的样本才可以达到相同地泛化误差上界。



### Infinite Hypothesis Space

本节研究无限维假设空间的泛化误差上界问题，此时需要引入VC维作为分析的工具。

为了将泛化误差上界VC维联系起来，首先需要将其和增长函数联系起来，其中需要用到以下的引理，将泛化误差转化为两个训练集上的经验损失之差，该技巧称为 Ghost Dataset 

待分析的数据集为 $D$, 而假设有一个独立同分布的数据集 $D'$， 可以得到如下结论，


$$
P( \sup_{h \in \mathcal{H}} R(h) > \hat R_D(h) + \epsilon  ) \le 2 P(\sup_{h \in \mathcal{H}} \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2}), \text{With } m \epsilon^2 \ge 2
$$


证明参考了 [STAT 598Y Statistical Learning Theory](https://www.stat.purdue.edu/~jianzhan/STAT598Y/NOTES/slt06.pdf)

首先令 $h$ 为左式取到上界的假设，对该假设进行证明，首先根据三角不等式，


$$
\begin{align}
I(R(h) > \hat R_D(h) + \epsilon )I( \hat R_{D'}(h) \le R (h) + \frac{\epsilon}{2})) &\le I( \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2})) \\
\end{align}
$$


对上述式子同时取期望可以得到，


$$
\begin{align}
P(R(h) > \hat R_D(h) + \epsilon )P( \hat R_{D'}(h) \le R (h) + \frac{\epsilon}{2})) &\le P( \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2})) \\
\end{align}
$$


而根据 Chebyshev 不等式，


$$
\begin{align}
P( \hat R_{D'}(h) \le R (h) + \frac{\epsilon}{2})) &\ge 1 - \frac{4 Var[\hat R_{D'}(h)]}{ \epsilon^2} \\
&= 1 - \frac{4 Var[R(h(x))]}{m \epsilon^2} \\
&= 1 - \frac{4 E[R^2(h(x))] - 4 E[R(h(x))]^2}{m \epsilon^2} \\
&\ge  1 - \frac{4 E[R(h(x))] - 4 E[R(h(x))]^2}{m \epsilon^2} \\ 
&= 1 - \frac{4 E[R(h(x))](1- E[R(h(x))])}{m \epsilon^2} ,\text{By } R(h(x)) \le 1\\
&\ge 1 -  \frac{1}{m \epsilon^2} \\
&\ge \frac{1}{2} , \text{By } m \epsilon^2 \ge 2
\end{align}
$$


代入就可以得到最终引理的形式，进而得到泛化误差界的结论，


$$
\begin{align}
P( \sup_{h \in \mathcal{H}} R(h) > \hat R_D(h) + \epsilon  ) &= P(  R(h) > \hat R_D(h) + \epsilon  ) ,\text{With Given } h \\
&\le  2 P(R(h) > \hat R_D(h) + \epsilon )P( \hat R_{D'}(h) \le R (h) + \frac{\epsilon}{2})) \\
&\le 2 P( \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2})) \\ 
&\le 2P(\sup_{h \in \mathcal{H}} \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2}) \\
&\le 2 \sum_{h \in \mathcal{H}}P( \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2}) \\
&= 2 \Pi(2m) P( \hat R_{D'}(h) > \hat R_D (h) + \frac{\epsilon}{2}) \\
&\le 2 \Pi(2m) \exp(-\frac{m\epsilon^2}{2}) ,\text{By Hoeffding Inequality} \\
&\le 2 (\frac{2em}{d})^d \exp(-\frac{m\epsilon^2}{2}) ,\text{By Sauser's Lemma}\\ 

\end{align}
$$


据此可以得到该问题的泛化误差界满足，


$$
P(R(h) \le  \hat R(h) + \sqrt{\frac{2 d \log \frac{2em}{d} + 2 \log \frac{2}{\delta}}{m}}) \ge  1- \delta ,\forall h \in \mathcal{H} ,\\
$$


值得注意的是，上述的证明实际上蕴含了有限维假设空间的情况，

此时增长函数有界，也即 $\Pi(2m) \le d$ , 可以得到收敛率为 $O(\sqrt \frac{1}{m})$

而对于无限维的情况，得到的收敛率为 $O(\sqrt{\frac{\log m}{m}})$ , 也意味着无限维假设空间中的学习问题更为复杂。



