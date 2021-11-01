---
title: '概率不等式'
toc: true
excerpt_separator: <!--more-->
tags:	
  - 统计机器学习
---



总结机器学习中常用的概率不等式的证明，包括Chevnoff，Hoeffding，Bennett，Bernstein不等式等，并且介绍次高斯变量相关的不等式和用随机投影进行数据降维的方法。

<!--more-->



## Chevnoff‘s Inequality

首先由著名的Markov不等式出发，


$$
\begin{align}
I[Y \ge t] &\le \frac{Y}{t} \\
P(Y \ge t) &\le \frac{EY}{t} \\
P(\exp \lambda Y \ge \exp \lambda t) &\le \frac{E\exp \lambda Y}{\exp \lambda t} \\
\end{align}
$$


为了寻求右端最紧的界，令$\phi(\lambda) = \log E \exp \lambda Y$ , 也即对数矩母函数，并且利用其共轭函数$\phi^{\star}(t) = \sup[\lambda t- \phi(\lambda)]$ ，可以得到Chevnoff不等式，


$$
P(Y \ge t) \le \exp( - \phi^{\star}(t))
$$


## Hoeffding Inequality

Hoeffding不等式通常在机器学习中用于泛化误差上界的推导，针对一系列独立有界的随机变量，$X_i \in [a_i,b_i]$, 有


$$
\begin{align}
P(S \le t) &\le \exp(-\frac{2t^2}{\sum_{i=1}^N (b_i -a_i)^2}) \\
\text{With }S &= \sum_{i=1}^N X_i - EX_i
\end{align}
$$
仅需要在$EX=0$的情况下证明即可，对于每一个$X_i$,利用其有界性，


$$
\begin{align}
X - \frac{a + b}{2}  &\le \frac{b-a}{2},\text{With } X \in [a,b] \\
E_g[(X - \frac{a + b}{2})^2] &\le (\frac{b-a}{2})^2,\text{With } g(X) = \frac{\exp(\lambda X)}{E\exp\lambda X} \\
E_g[X^2] - (E_g[X])^2 &\le   (\frac{b-a}{2})^2 \\
Var_g[X] &\le  (\frac{b-a}{2})^2 \\
\text{With }Var_g[X] &= \frac{E X^2 \exp \lambda X}{E \exp \lambda X} - (\frac{E X \exp \lambda X}{E \exp \lambda X})^2
\end{align}
$$


而对于$\phi(X)$, 对其求导可以得到，


$$
\begin{align}
\phi'(\lambda) &= \frac{E X \exp \lambda X}{E \exp \lambda X} = E_g[X]\\
\phi''(\lambda) &= \frac{E X^2 \exp \lambda X}{E \exp \lambda X} - (\frac{E X \exp \lambda X}{E \exp \lambda X})^2 = Var_g[X] 
\end{align}
$$


代入$\lambda=0$处进行Taylor展开可以得到，


$$
\begin{align}
\phi(\lambda) &= \phi(0) + \phi'(0) \lambda + \phi''(0)\frac{\lambda^2}{2} \\
&\le \frac{\lambda^2}{2}Var_g[X]  \\
&\le \frac{\lambda^2(b-a)^2}{8}
\end{align}
$$


又根据$X_i$的独立性假设，


$$
\begin{align}
\phi_S(\lambda) &= \log E \exp \lambda S \\
&= \log E \exp \lambda \sum_{i=1}^N X_i \\
&=\sum_{i=1}^N \log E \exp \lambda x_i \\
&= \sum_{i=1}^N \phi_{X_i}(\lambda)
\end{align}
$$



据此可以得到Hoeffding不等式，

$$
\begin{align}
P(S \ge t) &\le \exp \inf (\phi_S(\lambda)- \lambda t) \\
&\le  \exp \inf (\sum_{i=1}^N\phi_{X_i}(\lambda ) - \lambda t) \\
&\le \exp \inf (\frac{\lambda^2 \sum_{i=1}^N (b_i - a_i)^2}{8}  - \lambda t) \\
&= \exp (-\frac{2t^2}{\sum_{i=1}^N (b_i-a_i)^2}) \text{,With } \lambda = \frac{4t}{\sum_{i=1}^N(b_i-a_i)^2}
\end{align}
$$


## Bennett‘s Inequality

Hoeffding多用于有界独立变量的情况，而对于仅有上界而无下界的情况，也即$X_i \le b$，可以使用Bennett不等式，


$$
\begin{align}
P(S \ge t) & \le \exp(-\frac{v}{b^2}h(\frac{bt}{v}))\\
\text{With } h(x) &= (1+x) \log (1+x) - x, v = \sum_{i=1}^NE[X_i^2],S = \sum_{i=1}^N X_i- EX_i
\end{align}
$$


不失一般性，考虑对$b=1$的情况证明，对于$b \ne 1$的情况只需做个变换即可，


$$
P(S \ge t) \le \exp(t-(t+v) \log\frac{v+t}{v})
$$


其实只需要证明下式就可以证明该不等式，


$$
\begin{align}
\phi_S(\lambda) & \le v \varphi(\lambda ), \text{With } \varphi(x) = \exp x - x-1 \\
P(S \ge t) &\le \exp \inf \phi_S(\lambda) - \lambda t  \\
&= \exp \inf v \exp \lambda - \lambda -1-\lambda t \\
&= \exp(t-(t+v) \log\frac{v+t}{v}),\text{With } \lambda = \log \frac{v+t}{v}
\end{align}
$$
下面证明关键的不等式，$\phi_S(\lambda)  \le v \varphi(\lambda )$



首先我们知道$g(x) = \varphi(x) / x^2$ 为单调函数，该单调性可以根据Taylor展开得到，因此


$$
\begin{align}
g(\lambda X_i) & \le \lambda, \text{With } X_i \le 1 \\
\varphi(\lambda X_i) &\le X_i^2 \varphi(\lambda) \\
E[\exp \lambda X_i - \lambda X_i -1] &\le E[X_i^2] \varphi(\lambda) \\
E[\exp \lambda X_i] &\le E[X_i^2] \varphi(\lambda) + \lambda E[X_i] + 1 
\end{align}
$$


据此可以得到，
$$
\begin{align}
\phi_S(\lambda) &= \log E \exp (\lambda \sum_{i=1}^N X_i - EX_i)\\
&= \sum_{i=1}^N [\log E \exp \lambda X_i - \lambda EX_i], \text{By independency of } X_i \\
&\le \sum_{i=1}^N \log (E X_i^2\varphi(\lambda )+ \lambda E X_i +1)-\lambda \sum_{i=1}^N EX_i \\
&= n \sum_{i=1}^N \frac{1}{n} \log (E X_i^2\varphi(\lambda )+ \lambda E X_i +1)-\lambda \sum_{i=1}^N EX_i \\
&\le n \log \sum_{i=1}^N \frac{1}{n}[E X_i^2\varphi(\lambda )+ \lambda E X_i +1] - \lambda \sum_{i=1}^NEX_i , \text{By Concavity of } \log(x)\\
&\le v \varphi(\lambda), \text{By } \log(1+x) \le x
\end{align}
$$


由此我们证得了Bennett不等式，此处再次总结如下，


$$
P(S \ge t) \le \exp(-\frac{v}{b^2}h(\frac{bt}{v}))
$$




## Bernstein's Inequality



Bernstein不等式在Bennett不等式的基础上用一个多项式$g(x)$逼近$h(x)$，和Taylor展开密切相关，


$$
\begin{align}
h(x) \ge g(x) \\
\text{With }g(x) &= \frac{x^2}{2(1+\frac{x}{3})} \\
h(x) &= (1+x)\log (1+x) - x
\end{align}
$$


对两个函数同时求导即可以得到，


$$
\begin{align}
h(0) &= g(0) = 0 \\
h'(0) &= g'(0) = 0 \\
h''(x) &= \frac{1}{1+x} \\
g''(x) &= \frac{27}{(x+3)^3} \\
\end{align}
$$


可以归纳得到，


$$
\begin{align}
h^{(n)}(x) &\ge g^{(n)}(x) \\
\text{Then } h(x) &\ge g(x), \text{By Taylor's Expansion}
\end{align}
$$


利用该不等式代入就可以得到Bernstein不等式的形式，


$$
P(S \ge t) \le \exp(- \frac{t^2}{2(v + \frac{bt}{3})})
$$



## Inequality With Sub-Guassian 

下面给出关于次高斯分布的一些不等式，其中次高斯分布定义如下，


$$
E[\exp \lambda X] \le \exp  \frac{\lambda^2 \sigma^2}{2} 
$$


其中右端项即为高斯分布$\mathcal{N}(0,\sigma^2)$的矩母函数。



首先我们可以得到次高斯分布的性质，


$$
\begin{align}
\text{If }X &\sim G(\sigma^2) \\
\text{Then } E[X] &= 0, Var[X] \le \sigma^2 
\end{align}
$$
证明中用到了Taylor展开和极限性质，


$$
\begin{align}
E[\exp \lambda X] &\le \exp \frac{\lambda^2 \sigma^2}{2} \\
\lambda EX + \frac{\lambda^2}{2} EX^2 &\le \frac{\lambda^2\sigma^2}{2} + o(\lambda^2) \\
\end{align}
$$


令$\lambda \rightarrow 0$可以得到要证明的结论，


$$
E[X] = 0, Var[X] \le \sigma^2
$$




对于次高斯分布可以得到其对应的概率不等式，


$$
\begin{align}
P(X \ge t) & \le \frac{E \exp \lambda X}{\exp \lambda t} \\
& \le \inf\exp(\frac{\lambda^2\sigma^2}{2} - \lambda t) \\
& = \exp(-\frac{t^2}{2\sigma^2}) ,\text{With } \lambda = \frac{t}{\sigma^2}
\end{align}
$$


上述也可以看作对于$\vert X \vert$的不等式，


$$
P(\vert X\vert \ge t) \le 2 \exp(-\frac{t^2}{\sigma^2})
$$


对于次高斯分布的$p$阶矩，也可以得到相应的不等式，


$$
\begin{align}
E\vert X \vert^p &= \int P(\vert X \vert^p \ge t) dt \\
&=\int p t^{p-1} P(\vert X \vert \ge t) dt \\
& \le \int 2p t^{p-1} \exp(-\frac{t^2}{2\sigma^2} ) dt \\
& = p(2\sigma)^{p/2} \int u^{p/2-1} e^{-u} du ,\text{Let } u = \frac{t^2}{2\sigma^2} \\
&= p(2v)^{p/2} \Gamma(p/2),\text{Let } v = \sigma^2
\end{align}
$$


当$q=2p$的时候可以得到，


$$
E\vert X \vert ^{2q} \le 2(2v)^q  q! \le \frac{q!}{2}(4v)^q
$$



## New Bernstein's Inequality

观察到上述次高斯分布的结论，我们通常可以类似地假设


$$
\begin{align}
\sum_{i=1}^NE[X_i^2] &\le \sigma^2 \\
\sum_{i=1}^N E[X_i]_+^q &\le \frac{q! \sigma^2}{2} C^{q-2}, \exists C>0,\forall q\ge 3
\end{align}
$$


据此可以得到另一个新的Bernste不等式，


$$
\begin{align}
\phi_S(\lambda) &= \log E \exp (\lambda \sum_{i=1}^N X_i - EX_i)\\
&= \sum_{i=1}^N [\log E \exp \lambda X_i - \lambda EX_i], \text{By independency of } X_i \\
&\le \sum_{i=1}^N E [\exp \lambda X_i - \lambda X_i -1] , \text{By} \log(x) \le x+1 \\
&\le\sum_{i=1}^N E [\frac{\lambda^2 X_i^2}{2}], \text{By } e^x - x-1 \le \frac{x^2}{2}  \\
&\le \sum_{i=1}^N E [\frac{\lambda^2 X_i^2}{2}] + \sum_{i=1}^N  \sum_{q=3}^{\infty} \frac{\lambda^q}{q!}E[X_i]_+^q \\
&\le \frac{\lambda^2 \sigma^2}{2} + \sum_{q=3}^{\infty} \frac{\lambda^q}{q!} \frac{q! \sigma^2}{2} C^{q-2} \\
&= \frac{\lambda^2 \sigma^2}{2}\sum_{q=2}^{\infty} (\lambda C)^{q-2} \\
&= \frac{\lambda^2 \sigma^2}{2}\sum_{q=0}^{\infty} (\lambda C)^{q} \\
&= \frac{\lambda^2 \sigma^2}{2(1-\lambda C)}, \text{If }\lambda C < 1 
\end{align}
$$



计算其共轭函数，


$$
\begin{align}
P(S \ge t) &\le \exp \inf \phi_S(\lambda )- \lambda t \\
&= \exp \inf \frac{\lambda^2 \sigma^2}{2(1-\lambda C)} - \lambda t \\
&= \exp \inf \frac{\sigma^2}{C^2} [ (\frac{1}{2}+\frac{tC}{\sigma^2})u+\frac{1}{2u}- 1-\frac{tC}{\sigma^2}],\text{Let } u = 1-\lambda C \\
&= \exp \frac{\sigma^2}{C^2}(\sqrt{1+\frac{2tC}{\sigma^2}}- 1-\frac{tC}{\sigma^2}) \\
&= \exp (-\frac{\sigma^2}{C^2} h (\frac{t}{\sigma^2})), \text{Let } h(x)= 1+x -\sqrt{1+2x} 
\end{align}
$$


再利用如下不等式，可以简单验证，可以得到，


$$
\begin{align}
h(x) &\ge g(x), \text{Let } g(x) = \frac{x^2}{2(1+x)} \\
P(S \ge t) &\le \exp (-\frac{\sigma^2}{C^2} h (\frac{t}{\sigma^2})) \\
&\le \exp (-\frac{\sigma^2}{C^2} g (\frac{t}{\sigma^2})) \\
&\le \exp (- \frac{t^2}{2C^2(\sigma^2+t)})
\end{align}
$$
该形式与另一个Bernstein不等式也有相近之处。



## Sophisticate Projection

本节利用概率不等式研究随机投影用于降维，给定一个$p$维向量$u$,可以证明使用一个从$i.i.d$ 的标准高斯分布抽样出来的$d \times p$维随机矩阵作用在$u$上得到的新的向量可以起到降维的效果，并且在距离度量上保持较好的性质。


$$
\begin{align}
\text{Let } v &= \frac{Ru}{\sqrt d} \\
\text{Then We Have }
E \Vert v \Vert_2 &= \Vert u \Vert_2 \\
\text{And }P(\frac{\Vert v \Vert_2-\Vert u \Vert_2}{\Vert u \Vert_2} > \epsilon) & \le \exp(-\frac{d(\epsilon^2-\epsilon^3)}{4}) \\

\end{align}
$$




证明的关键在于证明下面定义的随机变量满足$\chi^2$分布，


$$
X \sim \chi^2(d), X = \frac{d\Vert v \Vert_2^2}{\Vert u \Vert_2^2}
$$


对$R$按行分块，


$$
\begin{align}
X &=  \frac{d\Vert v \Vert_2^2}{\Vert u \Vert_2^2} \\
&= \frac{\Vert Ru \Vert_2^2}{\Vert u \Vert_2^2} \\
&= \sum_{i=1}^d (\frac{R_i u}{\Vert u \Vert_2})^2 \\
&\sim \chi^2(d) ,\text{With } R_i u{\Vert u \Vert_2} \sim \mathcal{N}(0,1) 
\end{align}
$$


因此我们有$E[X] = d$，则可以得到进行随机投影后期望距离保持不变，也即，


$$
E \Vert v \Vert_2 = \Vert u \Vert_2
$$


我们更关心随机投影后是否能以较高的概率保持距离，因此利用概率不等式，


$$
\begin{align}
P(\frac{\Vert v \Vert_2-\Vert u \Vert_2}{\Vert u \Vert_2} > \epsilon) &= P(X \ge d (1+\epsilon)) \\
& \le \inf \frac{(1-2\lambda)^{-1/2}}{\exp (1+\epsilon) \lambda} ,\text{With } E[\exp \lambda X] = (1-2\lambda)^{-1/2}  \\
&=((1+\epsilon) \exp(-\epsilon))^{d/2} ,\text{With } 1-2\lambda  = \frac{1}{1+\epsilon} \\
&\le \exp(-\frac{d(\epsilon^2-\epsilon^3)}{4}) ,\text{With } \log x \le x - \frac{x^2}{2} + \frac{x^3}{2}  \\
\end{align}
$$


因此可以使用随机投影进行数据降维，该方法相较于PCA等方法更为简单。

关于PCA可以参见 [特征值不等式与最佳低秩逼近](https://truenobility303.github.io/Low-Rank-Approximation/)

