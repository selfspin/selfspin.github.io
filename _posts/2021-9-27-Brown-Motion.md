---
title: '布朗运动'
toc: true
excerpt_separator: <!--more-->
tags: 	
  - 随机过程
---

布朗运动的相关总结。布朗运动通常用来研究分子的无规律运动，模拟股票市场的波动情况等。本文主要包含了布朗运动的数字特征、样本轨道特征、布朗桥等基于布朗运动的重要随机过程、布朗运动的最大值和首中时等，不包括与随机积分相关的内容。



<!--more-->

## Definition and Properties

布朗运动的定义和性质

* $B_0  = 0$
* 增量平稳，$B_{t+s}-B_s \sim B_t$ 
* 增量独立，$B_{t+s} - B_s \perp B_t$
* $B_t \sim N(0,t)$

根据布朗运动的定义，可以计算其数字特征，包括均值，方差，协方差等。

* $E[B_t] = 0, Var[B_t] = t$
* 不妨设$s<t$, $Cov[B_t, B_s] = E[B_t B_s] = E[(B_t-B_s+B_s)B_s] = E[B_s]^2 = Var[s]=s$ 
* 一般地，$Cov[B_t,B_s] = \min(s,t)$

---

更一般地来讲，布朗运动本质上是一种正态过程。

正态过程定义为：任意有限维联合分布都是联合正态分布的随机过程，根据矩母函数，也可以定义为对于联合分布的任意线性组合为正态随机变量的随机过程。

对于一个正态过程，若其均值和协方差都与布朗运动相等，也即$E[B_t] = 0, Cov[B_s,B_t] = \min(s,t)$, 可以证明其为布朗运动。

证明中只需逐条验证布朗运动的定义。

$B(0) =0, B_t \sim (0,t)$ 显然成立。

对于增量平稳性质，由于正态分布由其期望和方差唯一决定，根据条件计算：

$E[B_{t+s} - B_s]  = t, Var[B_{t+s}-B_s] =  t$， 因此，$B_{t+s} - B_s \sim N(0,t)$

对于增量独立性质，由于正态分布不相关等价于独立，计算协方差：

$Cov[B_{t+s}-B_t, B_t] = Cov(B_{t+s},B_t) - Cov(B_{t},B_t)=0$ ,因此，$B_{t+s} - B_s \perp B_s$

## Simple Transformation

在布朗运动的基础上，经过简单的时空变换，还可以定义其他形式的布朗运动，验证其满足均值和协方差即可。

* $x_t =B_{t-t_0} - B_{t_0}$
* $x_t = tB_{\frac{1}{t}}$
* $x_t = \frac{1}{\sqrt{c}} B_{ct}$ 

首先上述过程均为布朗运动这个正态过程的线性组合，因而也为正态过程。根据正态过程与布朗运动的关系，计算可以得到，$E[x_t] = 0, Cov[x_t,x_s] = \min (s,t)$ ，因此根据上述结论，$x_t \sim B_t$

## Conditional Density

观测到布朗运动的部分条件之后，可以推断布朗运动的其余的条件概率。

例如，已知$B_s$想要推断$B_{s+t}$称为预测（Forecasting），已知$B_{s+t}$想要推断$B_s$称为倒推（Backcasting），本质上都是计算联合正态分布的条件概率，使用Schur补便可推出联合正态分布的条件分布，此处直接给出计算结果

* Forecasting：$B_{s+t} \vert b_s \sim (b_s, t)$
* Backcasting：$B_s \vert b_{s+t} \sim (\frac{b_{s+t}s}{t+s}, \frac{ts}{t+s})$

更一般地，已知布朗运动的过去和未来$(t_1,x_1),(t_2,x_2)$，可以推断布朗运动的现在$x_t \sim N(x_1 + \frac{(x_2-x_1)(t-t_1)}{t_2-t_1}, \frac{(t_2-t)(t-t_1)}{t_2-t_1})$

## Sample Path

本节研究布朗运动的样本轨道。

布朗运动可以看作简单随机游动的极限函数，简单随机游动指的是在数轴上从原点出发，每个时间点等概率地向左或者向右走一个单位长度$h$的随机过程。将$[0,t]$等分为$n$份，令$n \rightarrow \infty$,  游动的距离$X_t = \sum_{i=1}^n Z_i$ 

显然简单随机游动具有平稳和独立的增量，而根据中心极限定理，$X_t \rightarrow N(\mu,\sigma^2)$,下面求其期望和方差即可。

$E[X_t] = \sum_{i=1}^n E[Z_i] = 0, Var[X_t] = \sum_{i=1}^n Var[Z_i] = nh^2 = t,\text{ Let } h = \sqrt{\frac{t}{n}}$ 

也即只要选择特定的$h$,可以用简单随机游动逼近布朗运动。

---

有意思的是，布朗运动的样本轨道实际上定义了一个几乎处处连续但却无处不可导的函数。下面证明：

证明的本质用到的是概率极限定理，关于概率极限定理，可以参见 [假设检验](https://truenobility303.github.io/Hypothesis-Testing/) 中的基础知识部分。

对于其连续性，因为$B_{t+h} - B_t \sim N(0,h)$，当$h \rightarrow 0$的时候，$B_{t+h} \rightarrow B_t$  也可以使用Markov概率不等式更加严谨地证明。

对于其可微性，因为$\frac{B_{t+h} - B_t}{h} \sim N(0,\frac{1}{h})$ ,  当$h \rightarrow 0$的时候, 该正态分布方差趋近于无穷，因此其导数的极限不存在。

## Variations of Brown Motion

本节定义几个布朗运动的变形，在后面这些变形可能扮演者重要的作用

**Reflected Brown Motion**

反射布朗运动，定义为$\vert B_t \vert$，下面计算其期望、方差、分布函数等。

$E[ \vert B_t \vert] = \int_{0}^{\infty} \vert x \vert f(x) dx=  \sqrt\frac{2 t}{\pi}$

$Var[\vert B_t \vert] = E[B_t^2] - E[\vert B_t \vert]^2 = t - \frac{2t}{\pi}$

$P(\vert B_t \vert \le x) = 1-2P(B_t > x) = 2\Phi(\frac{x}{\sqrt{t}})-1$

其中,$\phi(x)$为标准正态的分布函数。

**Geometric Brown Motion**

几何布朗运动，定义为$\exp(a B_t)$,下面同样计算其数字特征,用到了正态分布的矩母函数。

$E[\exp(a B_t)]  = \exp(\frac{a^2 t}{2})$ ,由于$a B_t \sim N(0,\alpha^2 t)$

$Var[\exp(a B_t)] = E[\exp(2 a B_t)] - E[\exp(aB_t)]^2 = \exp(2a^2t) - \exp(a^2t)$

$P( \exp(a B_t) \le x) = \Phi(\frac{\ln a}{a \sqrt{t}})$,对分布函数求导可以得到密度函数。

**Brown Bridge**

布朗桥，定义为$x_t = B_t - tB_1,\text{with }t \in [0,1],B_t\sim \text{Brown(t)}$

显然，该过程也是一个正态过程，计算其期望和协方差。

$E[x_t] = E[B_t] - t E[B_1] = 0$

设$s<t$, $Cov[x_t,x_s] = Cov[B_t - tB_1,B_s- sB_1] = s-ts$

特别地，$Var[x_t]  = t(1-t)$，回顾条件概率一节，我们发现$x_t \sim B_t \vert B_1=0$.

本质上，布朗桥为给定起点和终点位置$B_0=0,B_1=0$的前提下，中间其他时刻$t$的布朗运动。

**Integrated Brown Motion**

积分布朗运动，定义为$x_t = \int_0^t B_s ds$

将积分写成部分和的极限，正态分布的部分和为正态分布，而正态序列的极限分布仍然为正态分布，因此积分布朗运动也为正态过程。

计算其均值和方差，则可以确定这个正态过程。

均值是显然的，$E[x_t] = 0$，下面计算器协方差，假设$s <t$，根据期望和积分的可交换性和重积分的可交换性：
$$
\begin{align}
Cov[X_tX_s] &= E[\int_0^s B_udu\int_0^t B_vdv ] \\
&=\int_0^s du \int_0^t dv  Cov[B_u,B_v] \\
&= \int_0^s du \int_0^u  vdv  +\int_0^s du \int_u^t udv \\
& = \frac{ts^2}{2} - \frac{s^3}{2}
\end{align}
$$



## Maximum Value and Hitting Time

本节探究布朗运动的最大值和首中时等分布

**Maximum Value**

布朗运动的最大值定义为：$M_t = \sup B_t$,下面求解$M_t$的分布函数，主要利用了布朗运动中常见的反射技巧。

首先，根据$B_t$与$x$的大小关系将概率$P(M_t >x)$分解为两项：
$$
\begin{align}
P(M_t > x) = P(B_t > x) +P(B_t \le x, M_t>x)
\end{align}
$$
对于第一项，根据$B_t \sim N(0,t)$可得，下面考虑第二项：

根据样本轨道（Sample Path）一节中的结论，我们知道其为连续函数，根据零点定理，布朗运动一定存在某点首次触及$x$，记其时间为$T_x$，由于$T_x$之后的状态与之前的状态无关，因此我们对$T_x$之后的轨迹关于$x$进行一个反射，由于布朗运动的对称性，得到的也是一个布朗运动。本质上，$T_x$是布朗运动的停时（Stopping Time）

反射前，$B_t \le x$, 则反射后，$B_t > x$ 

因此，$P(B_t \le x, M_t > x) = P(B_t > x)$

综上，$P(M_t > x) = 2P(B_t > x) = P(\vert B_t \vert > x)$

也即有：$ M_t \sim \vert B_t \vert $, 即最大值的分布与反射布朗运动的分布相同。

**Hitting Time**

首中时，其实就是上面定义的$T_x=\inf B_t = x$ ,也即布朗运动首次碰到$x$的时刻。

同样根据连续函数的零点存在定理，$P(T_x \le t)  = P(M_t > x) = 2(1-\Phi(\frac{x}{\sqrt{t}}))$

显然，$P(T_a < \infty) = 1$

求导后得到密度函数，积分后可以证明：$E[T_a] = \infty$

联系马尔可夫链中的状态定义，可以参见 [马尔可夫链](https://truenobility303.github.io/Markov-Chain/)

对于任意的位置$a$, 布朗运动的轨道在有限时间内到达$a$的概率都为1，但期望到达时间却为$\infty$,对应于马尔可夫链中的零常返态。

**Minimum Value**

类似于，布朗运动最大值$M_t$的分布，其最小值$m_t$的分布，可以根据对称性得到。

对于$0$做个反射可以得到，$P(m_t \le x) = P(M_t \ge -x) = 2\Phi(\frac{x}{\sqrt t})$

**Joint Distribution of $(B_t,M_t)$**

下面我们考虑$(B_t,M_t)$的联合分布,利用类似的技巧构造出包含$M_t >y$的概率项。
$$
P(B_t \le x, M_t \le y) = P(B_t \le x) -  P(B_t \le x, M_t > y)
$$

对停时$T_y$以后的布朗运动进行反射变换，可以得到，反射后,$B_t > 2 y -x$

因此，$P(B_t \le x, M_t >y) = P(B_t > 2y-x)$

综上，$P(B_t \le x, M_t \le y) = \Phi(\frac{x}{\sqrt t}) - 1 + \Phi(\frac{2y-x}{\sqrt t})$

**Absorbed Brown Motion**

根据$(B_t,M_t)$的联合分布，可以用来研究吸收布朗运动。

吸收布朗运动指的是，当$B_t$触及$x$之后，$B_t$保持在$x$的水平，也即被吸收了。

记吸收布朗运动为$Z_t$，为了研究其分布，下面计算$P(Z_t \le y)$

$y=x$时，$P(Z_t \le y) = P(Z_t \le x) = P(T_x <= t)$，转化为首中时$T_x$的分布。

对于$y< x$的情况，其分布可以用$(B_t,M_t)$的联合分布表示: 
$$
P(Z_t \le y) = P(B_t \le y, M_t \le x)
$$
上式的含义是，该布朗运动不仅要在$t$时刻到达给定的区域，在过程中还不能触及吸收态$x$.
