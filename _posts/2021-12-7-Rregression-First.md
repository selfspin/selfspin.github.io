---
title: '回归分析上'
toc: true
excerpt_separator: <!--more-->
tags: 
  - 统计机器学习

---



线性回归入门级总结，包括线性回归模型的基本性质，假设检验和区间估计，变量选择和准则，以及对于异方差和自相关等现象的解决。



<!--more-->

## Linear Regression Model  

给定模型真实参数$\theta$， 统计推断的目的是根据数据$X$估计出参数$\hat \theta$.

常用的估计方法有：

* 矩估计，基于的思想是经验分布函数收敛到真实分布函数，其证明见 [Blog](https://truenobility303.github.io/MLE/)
* 极大似然估计，选取$\hat \theta$使得似然函数$L(\theta)$或者对数似然函数$\log L(\theta)$最大化的参数, 极大似然估计具有很多性质，见 [Blog](https://truenobility303.github.io/MLE/)

极大似然估计通常可以基于求导求得极值，例如给定i.i.d的正态分布样本$X_1,X_2,..X_n$,容易验证其均值和方差的极大似然估计即为样本均值和样本方差，也即 $\hat \mu = \bar X, \hat \sigma^2  = \frac{1}{n}(X-\bar X)^2$

---

估计出估计量$\hat \theta$之后，需要选择评判标准评判估计的好坏，常用的标准有：

* 偏差，$\text{Bias} = E[\hat \theta] - \theta$ ,表示估计量的期望与真实参数之间的差距
* 方差，$\text{Var} = E[\hat \theta - E \hat \theta]^2$ ,表示估计出来的参数$\hat \theta$的波动性

偏差和方差都是越小越好，但通常两者不能同时满足，可以使用均方误差MSE（Mean Square Error）来衡量，其中MSE定义为$\text{MSE} = E[\hat \theta - \theta]^2$ ,MSE满足偏差-方差分解：


$$
\begin{align}
\text{MSE} &= E[\hat \theta - \theta]^2 \\
&=E[\hat \theta - E \hat \theta + E \hat \theta -\theta]^2 \\
&= E[\hat \theta - E \hat \theta]^2 + [E \hat \theta -\theta]^2 + 2[E \hat \theta -\theta]E[\hat \theta - E \hat \theta] \\
&= \text{Var} + \text{Bias}^2
\end{align}
$$


---

线性回归模型假设，给定自变量$X$和因变量$y$，其满足$y= X \beta + \epsilon$,其中误差项$\epsilon$又满足独立零均值同方差的假设，也即
$$
\epsilon_i \sim i.i.d, E[\epsilon]=0,Var[\epsilon_i] = \sigma^2
$$
首先需要说明的是，该模型的假设是真实存在的，考虑多元联合正态分布，


$$
\begin{align}

\begin{pmatrix}
y \\
X 
\end{pmatrix}
\sim
\mathcal{N}

\begin{pmatrix}
\mu_y\\
\mu_x 
\end{pmatrix}
,
\begin{pmatrix}
\Sigma_{yy} & \Sigma_{xy} \\
\Sigma_{xy} & \Sigma_{xx} 
\end{pmatrix}

\end{align}
$$


根据联合正态分布的条件分布，可以通过Schur补的方式推出，给定$X$时$y$的条件分布满足


$$
\begin{align}
y \vert X &\sim \mathcal{N}(\mu,\Sigma) \\
\text{With } \mu &= u_y + \Sigma_{yx} \Sigma_{xx}^{-1} (X - \mu_x) \\
\Sigma &=  \Sigma_{yy} - \Sigma_{yx} \Sigma_{xx}^{-1} \Sigma_{xy}
\end{align}
$$


此时满足线性回归的模型假设，$y = X \beta + \epsilon$

---

线性回归可以用最小二乘方法解，也即希望最小化$\Vert X \hat \beta -y \Vert_2$.

使用求导求函数极值的方法可以得到，最小二乘解为，


$$
\hat \beta = (X^T X)^{-1} X^T y
$$


或者可以使用投影的角度思考，最小二乘解需要满足, 在$X$的空间上的投影相等，


$$
\begin{align}
\langle X, X \hat \beta \rangle &= \langle X, y \rangle \\
X^TX  \hat \beta  &= X^T y
\end{align}
$$


可见利用投影的角度思考也可以得到相同的结果。



## Main Properties

对于线性回归得到的参数$\hat \beta$,可以通过计算得到，


$$
\begin{align}
E[\hat \beta] &= E[(X^T X)^{-1} X^T (X \beta + \epsilon)] = \beta \\
Var[\hat \beta] &= Var[(X^T X)^{-1} X^T  \epsilon] = (X^TX)^{-1} \sigma^2 \\
\end{align}
$$


可以看到$\hat \beta$ 为真实参数$\beta$的无偏估计，且方差与$\sigma^2$相关。



我们关心$\hat \beta$的方差，而我们可以证明$\hat \beta$满足如下的Gauss-Markov定理，也即其为最小无偏估计(B.L.U.E, Best Least Unbiased Estimator)


$$
\begin{align}
&\forall  \eta \text{ s.t. } \Vert \eta  \Vert_2 =1, Var[\eta^T \tilde \beta] \ge Var[\eta^T \hat \beta] \\
\text{With } &E[\tilde \beta] = \beta, \tilde \beta  = W y\\
\end{align}
$$


首先$W$必须满足无偏性，


$$
\begin{align}
E[\tilde \beta] &= E[W(X \beta +\epsilon)] = E[WX \beta] = \beta  \\
\text{Only if }  WX &= I
\end{align}
$$


而对于最小方差的要求，等价于证明矩阵$WW^T - (X^TX)^{-1}$为半正定阵



令$D = W - (X^TX)^{-1}X^T$ , 可以得到，$DX = O$, 因此


$$
\begin{align}
W W^T &= (D + (X^TX)^{-1}X^T)(D + (X^TX)^{-1}X^T)^T \\
&= D D^T + (X^TX)^{-1} + 2 (X^TX)^{-1}X^T D^T \\
& = DD^T + (X^TX)^{-1}
\end{align}
$$


显然$DD^T$为半正定阵，因此$WW^T - (X^TX)^{-1}$也为半正定阵。

---



上面我们估计了参数$\hat \beta$，而模型中还有一个未知参数$\hat \sigma^2$需要估计，为了对其进行估计，首先需要对$\epsilon$ 的分布进行假设。

简单的假设是$\epsilon \sim \mathcal{N(0,\sigma^2)}$ ,这也符合多元正态的联合分布中的假设。



我们可以发现，在上述假设之下，基于$\epsilon$的似然函数对$\hat \beta, \hat \theta $做极大似然估计(MLE,Maximum Likelihood Estimate)，

写出对数似然函数为，



$$
\begin{align}

L(\beta,\sigma^2) &=  \log \prod_i \frac{1}{\sqrt{2 \pi \sigma^2}} \exp(-\frac{(y_i - X_i \beta)^2}{2 \sigma^2}) \\
\log L(\beta,\sigma^2) &= -\frac{1}{2 \sigma^2} \Vert y - X \beta \Vert_2^2 - \frac{n}{2} \log 2\pi - \frac{n}{2} \log \sigma^2 
\end{align}
$$



对于$\hat \beta$，需要最小化$ \Vert y - X \beta \Vert_2^2$ ,也即等价于最小二乘估计.



$$
\hat \beta = (X^TX)^{-1}X^T y
$$



对于$\hat \sigma^2$，对对数似然函数求导可以得到，



$$
\frac{\partial s}{ \partial \sigma^2} = -\frac{n}{2 \sigma^2} + \frac{1}{2\sigma^4} \sum_i ( y_i - X_i \beta )^2
$$



因此化简可以得到，$\sigma^2$的极大似然估计，

$$
\hat \sigma^2 = \frac{1}{n} \Vert y - X \hat \beta \Vert_2^2
$$


令$\hat \epsilon = y - X \hat \beta$, 则


$$
\hat \sigma^2 = \frac{1}{n} \hat \epsilon^T \hat \epsilon
$$

---

从投影的角度来理解最小二乘，令$H = X(X^TX)^{-1}X^T$, 并且对比$Y=  X \beta +\epsilon$ ,



$$
\begin{align}
y &= H y + (I- H) y \\
&= X(X^TX)^{-1}X^T y + (I-H)y \\
&=  X \hat \beta + \hat \epsilon \text{ (With  } \hat \epsilon = (I-H)y)
\end{align}
$$



也即残差项的估计$\hat \epsilon$ 即为投影的误差.

对于投影矩阵$H,I-H$，易验证其均为对称矩阵，也为幂等矩阵，其性质至关重要，且$HX = X,(I-H)X = O$



上面给出了对其方差的估计$\hat \sigma^2$，但实际上该估计是有偏的，由于



$$
\begin{align}

E[\hat \epsilon^T \hat \epsilon ] &= E[y^T (I-H)y] \\
&= E[\epsilon^T (I-H) \epsilon] \text{ (By } (I-H)X = 0)\\
&= E[tr(\epsilon^T (I-H) \epsilon)] \\
&= E[\epsilon^T \epsilon] tr(I-H) \\
&= (n-p) \sigma^2
\end{align}
$$



因此，需要将其修正为无偏估计，


$$
S^2 = \frac{1}{n-p} \hat \epsilon^T \hat \epsilon
$$


---



在假设了$\epsilon \sim \mathcal{N}(0,\sigma^2)$的前提后，可以求出估计量的分布，有以下的重要结论：


$$
\begin{align}

\hat \beta &\sim \mathcal{N}(\beta, (X^TX)^{-1} \sigma^2) \\
(n-p) \hat \sigma^2 &\sim \chi^2(n-p) \sigma^2\\
\hat \beta  &\perp \hat \sigma^2
\end{align}
$$


其中，$\hat \beta \sim \mathcal{N}(\beta, (X^TX)^{-1} \sigma^2)$ 是显然的，因为正态分布仅由其均值和方差决定。

而对于$\hat \sigma^2$ ,可以通过将幂等矩阵$I - H$进行正交对角化实现，


$$
\begin{align}
\hat \sigma^2  &= \epsilon^T (I-H) \epsilon \\
& = \epsilon^T Q^T \Lambda  Q\epsilon \\
\text{ With } \Lambda &= \text{diag}(1,1,1,...0,0,0) , tr(\Lambda) = n-p 
\end{align}
$$


可见，$\hat \sigma^2$为$n-p$个独立的零方差同均值正态随机变量的组合，因此$(n-p) \hat \sigma^2 \sim \chi^2(n-p) \sigma^2$



对于$\hat \beta  \perp \hat \sigma^2$ ，实际上可以证明$\hat \beta  \perp \hat \epsilon$ 

由于$\hat \beta, \hat \epsilon$ 都服从正态分布，因此只需验证其协方差为0，则不相关等价于独立



$$
\begin{align}
Cov[\hat \beta, \hat \epsilon] &= Cov[(X^TX)^{-1}X^T y, (I-H) y] \\
&= Cov[(X^TX)^{-1} X^T\epsilon, (I-H) \epsilon] \\
&= E[(X^TX)^{-1}X^T (I-H) ] \sigma^2 I_n \\
&= O \text{ (By } \langle I-H , X \rangle =0)
\end{align}
$$



因此我们给出了估计量$\hat \beta, \hat \sigma^2$的分布，并且证明了其独立性，该独立性质对于后续的假设检验的构建至关重要。



## Fitting Performance

考虑对线性回归的离差平方和做分解，首先显然有，



$$
\begin{align}
Y^T Y &= Y^T(H+I-H) Y \\
&=  Y^T H Y +Y^T (I-H)Y\\
&=  \hat Y^T \hat Y + \hat \epsilon^T \hat \epsilon^T
\end{align}
$$



由于我们更加关心无截距项无关的量，因此我们将总体减去均值作中心化得到如下的等式，

$$
\begin{align}
Y^T(I - \frac{1}{n}ee^T)Y = Y^T(H - \frac{1}{n}ee^T)Y +Y^T(I-H)Y \\
\end{align}
$$



其中左端项为总离差平方和，右端项被分解为可以被模型解释的部分，以及与残差估计相关的部分。



基于此，可以定义$R^2$作为回归效果的度量，



$$
R^2 = \frac{ (\hat Y - \bar Y)^T (\hat Y - \bar Y)}{(Y - \bar Y)^T (Y - \bar Y)}
$$
容易知道,$0 \le R^2 \le 1$, 且：

* $R^2=0$，此时$\hat Y = \bar Y$, 也即使用因变量的均值作为预测，相当于没有回归
* $R^2=1$， 此时$\hat Y = Y$ ,也即数据的所有值都被拟合出来了，回归效果很好

在上式中减去$\bar Y$的原因是这样做可以消去截距项的影响，截距项是在假设检验等中都不希望检验的量。



进一步考虑$R^2$的意义，对样本协方差$Cov[Y, \hat Y]$进行计算，并且写成矩阵的形式上式会更加好理解，分子分母同时约掉了相同的一项，



$$
\begin{align}
Corr^2[Y,\hat Y] &= \frac{Cov[Y,\hat Y]}{Var[Y] Var[\hat Y]} \\
&= \frac{[Y^T (I - \frac{1}{n}ee^T) (I - \frac{1}{n}ee^T)HY]^2}{[Y^T(I-\frac{1}{n}ee^T)Y][Y^T(H-\frac{1}{n}ee^T)Y] } \\
&= \frac{[Y^T (I - \frac{1}{n}ee^T) HY]^2}{[Y^T(I-\frac{1}{n}ee^T)Y][Y^T(H-\frac{1}{n}ee^T)Y] } \\
&= \frac{Y^T(H-\frac{1}{n}ee^T)Y}{Y^T(I-\frac{1}{n}ee^T)Y} \\
&= R^2
\end{align}
$$


因此，上式实际上说明了，

$$
\begin{align}
 Corr^2[Y, \hat Y] = R^2
\end{align}
$$



由于$R^2$ 随着自变量的增加而增加，越复杂的模型$R^2$越大，通常使用自由度调整后的R方，可以衡量你和效果和复杂度之间的关系，

$$
\begin{align}
R^2 &= 1-\frac{SSE}{SST} \\
\text{Adjusted }R^2 &= 1 - \frac{SSE / n-p}{SST / n-1} \\
\end{align}
$$




## Hypothesis Testing

我们关心模型是否有效，也即希望知道自变量和因变量是否相关，此希望对$\beta = [\beta_0,\beta_1,...\beta_{p-1}]$ 进行检验。

由于我们通常不在乎截距项，因此减去了$\bar Y$，此时相当于忽略了$\beta_0$

### Significance Test

这里我们仅考虑两种类型的假设检验，

* $\beta_0 = \beta_1 = \beta_2= ... =\beta_{p-1}= 0$ ,模型中所有的参数都无效，称为回归方程的显著性检验
* $\beta_n = 0$ ,模型中的某个参数无效，称为回归系数的显著性检验



当假设$\beta_1 = \beta_2= ... =\beta_{p-1}= 0$成立的时候，

样本因变量的值都为随机正态误差，也即，$Y \sim \epsilon$ , 因此根据独立同分布正态变量的样本方差的熟知结论，见 [Blog](https://truenobility303.github.io/Hypothesis-Testing/)


$$
\begin{align}
(Y - \bar Y)^T (Y - \bar Y) & = Y^T(I - \frac{1}{n} ee^T)Y \\
(Y - \bar Y)^T (Y - \bar Y) &\sim \chi^2(n-1) \\
\end{align}
$$



而考虑，$(\hat Y - \bar Y)^T (\hat Y - \bar Y)$, 类似于对$\hat \sigma^2$的分布的证明，易得到, 

$$
\begin{align}
(\hat Y - \bar Y)^T (\hat Y - \bar Y) &= Y^T(H - \frac{1}{n} ee^T) Y \\
(\hat Y - \bar Y)^T (\hat Y - \bar Y) &\sim \chi^2(p-1)
\end{align}
$$


类似对于$\hat \beta  \perp \hat \sigma^2$ 的证明


$$
\begin{align}
\text{Since } (H - \frac{1}{n} ee^T) (I - H) &= 0 \\
(\hat Y - \bar Y)^T (\hat Y - \bar Y) & \perp \hat \epsilon^T  \hat \epsilon
\end{align}
$$

其实这个独立性的证明不依赖于零假设，


$$
\begin{align}
SSR &= Y^T(H- \frac{1}{n}ee^T)Y \\
SSE &= Y^T(I-H) Y \\
Cov[(H- \frac{1}{n}ee^T)Y, (I-H) Y] &= Cov[(H- \frac{1}{n}ee^T) \epsilon, (I-H) \epsilon] \\
&= \sigma^2 (H- \frac{1}{n}ee^T)(I-H) =0
\end{align}
$$
对于该线性回归模型，满足正态性的假设，因此$Y$服从正态分布，而正态分布不相关等价于独立，因此SSR和SSE一定是相互独立的。本质上是由于两者前面的投影矩阵，正好投影到了不同的空间上面。



总而言之，可以构造F统计量，
$$
F = \frac{SSR / (p-1)}{SSE/ (n-p)} \sim F(p-1,n-p)
$$




---

自变量和隐变量相互独立的假设检验，根据$\hat \beta=0$做t检验 

对$\beta_n = 0$ 进行假设检验的时候，用到了联合正态分布的边缘分布，


$$
\begin{align}
\hat \beta & \sim \mathcal{N}(\beta , (X^TX)^{-1} \sigma^2) \\
\hat \beta_n & \sim \mathcal{N}(0, (X^TX)^{-1}_n \sigma^2) \\
\end{align}
$$


当$\sigma^2$已知的时候，可以使用z检验，但此时$\sigma^2$未知。

因此我们使用$\hat \sigma^2$进行嵌入，并且利用到了$\hat \beta  \perp \hat \sigma^2$的关系，构造出相应的t统计量，



$$
t = \frac{\hat \beta_n}{\sqrt{(X^TX)^{-1}_n \hat \sigma^2}} \sim t(n-p)
$$

### Confidence Interval

利用回归结果，可以对未知的数据点$x_{n+1}$对应的因变量$y_{n+1}$的值做预测，也即

$$
E[\hat y_{n+1}] = x_{n+1} \hat \beta
$$



但这样得到的是点估计，为了得到一个区间估计，我们计算$x_{n+1} \hat \beta$的分布，



$$
\begin{align}
\hat \beta &\sim \mathcal{N}(\beta, (X^TX)^{-1} \sigma^2 ) \\
x_{n+1} \hat \beta &\sim \mathcal{N}(x_{n+1} \beta, x_{n+1} (X^TX)^{-1} x_{n+1} \sigma^2)
\end{align}
$$



据此在$\sigma^2$已知的前提下，可以根据正态分布的$\alpha$分位数得到一个置信区间，称为Mean Response

在模型中，$\sigma^2$未知，此时需要利用估计量$\hat \sigma^2$进行嵌入，得到t统计量



$$
t = \frac{x_{n+1} \hat \beta }{\sqrt{x_{n+1} (X^TX)^{-1} x_{n+1} \hat \sigma^2}} \sim t(n-p)
$$


---

如果考虑$\epsilon$项，需要扩大置信区间，也即在$\sigma^2$已知的前提下为，称为Individual Response



$$
\begin{align}

\hat y_{n+1} = x_{n+1} \hat \beta +\hat \epsilon &\sim \mathcal{N}(x_{n+1} \beta, (1+x_{n+1} (X^TX)^{-1} x_{n+1}) \sigma^2  )
\end{align}
$$



利用$\hat \sigma^2$嵌入后得到t统计量为，

$$
t = \frac{x_{n+1} \hat \beta }{\sqrt{(1+x_{n+1} (X^TX)^{-1} x_{n+1}) \hat \sigma^2}} \sim t(n-p)
$$



### Extra Sum of Square Test

更通用的假设是检验部分系数$\beta=0$，此时考虑部分$\beta_n$线性相关，

可以考虑更广义的版本，也即$C \beta = 0,\text{rank}(C) = q$ ,也即模型实际上可以被缩减为$q$个系数表示



此时的最小二乘问题转化为带约束的最小二乘问题，使用Lagrange乘子法，同时根据$\hat \beta = (X^TX)^{-1}X^T Y$



$$
\begin{align}
\text{min } M &=(X \beta - Y)^T (X \beta- Y) + 2\lambda C \beta \\
\frac{\partial M}{\partial \beta} &=2X^T(X\beta -Y) + 2\lambda C^T = 0 \\
\beta &= (X^TX)^{-1} (X^T Y-\lambda C^T ) \\
0=C \beta &= C (X^TX)^{-1} (\lambda C^T - X^T Y) \\
\lambda &=  (C(X^TX)^{-1}C^T)^{-1} C\hat \beta \\
\beta &= \hat \beta - (X^TX)^{-1} C^T (C(X^TX)^{-1}C^T)^{-1} C\hat \beta \\
&= \hat \beta - A \hat \beta \\
\text{Where } A&= (X^TX)^{-1} C^T (C(X^TX)^{-1}C^T)^{-1} C
\end{align}
$$



计算在约束下的残差平方和$SSE$，





$$
\begin{align}
\text{SSE} &= \Vert Y - X \beta \Vert \\
&=  \Vert Y - X \hat \beta - XA \hat \beta \Vert\\
&= \Vert Y - X \hat \beta \Vert + \Vert XA \hat \beta \Vert + 2\text {Cross Item}
\end{align}
$$



下面计算上式留下的交叉项，



$$
\begin{align}
\text {Cross Item} &= Y^T XA \hat \beta - \hat \beta^T X^T XA \hat \beta\\ 
&= Y^T XA \hat \beta - Y^T X (X^TX)^{-1} X^TXA \hat \beta \\
&= Y^T XA \hat \beta - Y^T XA \hat \beta \\
&= 0
\end{align}
$$



上述计算看似繁琐，实际上交叉项为零的结论是显然的，
$$
\begin{align}
\text {Cross Item} &= \hat \beta^T (I-H) X A \hat \beta  =0,  \\
\text {With} (I-H)X &= O
\end{align}
$$


因此，与无约束的平方和SSE‘相减得到，

$$
SSE - SSE’ = \Vert XA \hat \beta \Vert
$$

考虑其分布，



$$
\begin{align}
\Vert XA \hat \beta \Vert &= \hat \beta^T A^T X^T XA \hat \beta \\
&= Y^T X (X^TX)^{-1} A^T  X^T XA (X^TX)^{-1}X^T Y \\
& = Y^T P Y \\
\text{Where  } P&= X (X^TX)^{-1} A^T  X^T XA (X^TX)^{-1}X^T
\end{align}
$$



上述的$P$为幂等矩阵，计算其自由度，反复利用$tr(AB) = tr(BA)$，得到，

$$
tr(P) = tr(I_q) =q
$$


构造F统计量，


$$
F = \frac{(SSE - SSE') / q}{ SSE' / (n-p)} \sim F(q,n-p)
$$

其中分子和分母的自由度已经算出，而独立性由于分子依赖于$\hat \beta$, 分母依赖于$\hat \epsilon$ ,根据之前的结论知道两者相互独立。



本质上，由于


$$
C \hat \beta \sim \mathcal{N}(C \beta,C (X^TX)^{-1}C^T)
$$


因此，
$$
\begin{align}
SSE  - SSE' &=  \Vert X A \hat \beta \Vert_2 \\
&= \hat \beta^T A^T X^T X A\hat \beta \\
&= \hat \beta^T C^T(C (X^TX)^{-1}C^T)^{-1} C \hat \beta \\
&\sim \chi^2(m)
\end{align}
$$



该检验被称为Extra Sum of Square检验方法，值得注意的是该检验方法是一种普遍情况，利用该检验可以简单地推出单个回归系数的假设检验和所有回归系数的假设检验。



对于单个回归系数的显著性检验，只需要令$C = [0,0,1,...,0]$ , 可以得到


$$
\begin{align}
\frac{(C \hat \beta^T) (C (X^T X^{-1})C^T)^{-1} C  \hat \beta }{SSE/ (n-p)} = \frac{\hat \beta_j^T \hat \beta_j}{S (X^TX)^{-1}_{jj}}  \sim F(1,n-p)
\end{align}
$$


而对于所有回归系数的显著性检验，只需要令$C=I$, 则可以得到，


$$
\begin{align}
\frac{\hat \beta (X^TX)^{-1} \hat \beta}{SSE / (n-p)} = \frac{SSR/(p-1)}{SSE/(n-p)} \sim F(p-1,n-p)
\end{align}
$$


本质上，采用其他的方法也可以推出上述的两个显著性检验，例如根据 $\hat \beta$的分布，


$$
\begin{align}
\hat \beta & \sim \mathcal{N}(0, (X^TX)^{-1}\sigma^2) \\
\hat \beta_j & \sim \mathcal{N}(0, (X^TX)^{-1}_{jj}\sigma^2) \\
\hat \beta^T(X^TX)^{-1} \hat \beta & \sim \chi^2(p) \sigma^2 \\
\end{align}
$$


如果在方差未知的情况下，采用样本方差嵌入进行估计，并且不关心与均值相关的项而作中心化操作，上述的正态分布将变为t分布，卡方分布将变为F分布，就可以得到类似的t检验和F检验的形式。因此，假设检验可能具有多种推导的方法，但可以发现不同的方法本质上总是在做同一件事情。

### Log Likelihood Ratio Test

另外一种检验方式是似然比检验，也即检验似然函数的差值，假设真实参数为$\beta$ ,可以使用卡方检验，令对数似然函数为$l$,则有

$$
2 l(\hat \beta) - 2l( \beta) \sim \chi^2(p)
$$

证明使用Taylor展开，


$$
\begin{align}
l(\beta) &= l(\hat \beta ) + l(\hat \beta)'(\beta  - \hat \beta) +\frac{1}{2} l''(\hat \beta ) (\beta  - \hat \beta)^2 \\
&= l(\hat \beta ) +\frac{1}{2}(\beta  - \hat \beta)^T l''(\hat \beta ) (\beta  - \hat \beta) \text{ (With } l'(\hat \beta)=0)\\

2 l(\hat \beta) - 2l( \beta) &=  -(\beta  - \hat \beta)^T l''(\hat \beta ) (\beta  - \hat \beta)  \\

\end{align}
$$



根据MLE的性质，
$$
\beta_i - \hat \beta_i  \sim \mathcal{N}(0,\frac{1}{I}), E[l''(\hat \beta )] = -I
$$


因此，可以得到，
$$
2 l(\hat \beta) - 2l( \beta) \sim \chi^2(p)
$$



上述的似然比检验在很多使用极大似然估计MLE的场合都可以用到，例如Logistic回归模型等。



## Examples and Applications 

考虑常见的一元情况，设


$$
Y =X \beta + \epsilon = \beta_0 + \beta_1 x + \epsilon
$$



此时将$\beta_0,\beta_1$分别考虑，则可以得到，



$$
\begin{align}
\hat \beta &\sim \mathcal{N}(\beta , (X^TX)^{-1} \sigma^2) \\

X^TX &= 

\begin{pmatrix}
n & n \bar x \\
n \bar x & \sum_i x_i^2
\end{pmatrix} \\

det(X^TX) &= n \sum_i x_i^2 - (n \bar x)^2  = n S_{xx}\\
\hat \beta_0 & \sim \mathcal{N}(\beta_0, \frac{\sigma^2 \sum_i x_i^2}{n S_{xx}}) \sim \mathcal{N}(\beta_0, \sigma^2(\frac{1}{n}+ \frac{\bar x^2}{S_{xx}})  \\
\hat \beta_1 & \sim \mathcal{N}(\beta_1, \frac{\sigma^2 }{S_{xx}}) \\

\end{align}
$$

上面给出了回归系数的分布，也即给出了回归系数的均值和方差，更进一步，还可以给出回归系数的协方差，


$$
\begin{align}
Cov[\hat \beta_0,\hat \beta_1] &= Cov[\bar y - \hat \beta_1 \bar x, \hat \beta_1] \\
&=Cov[-\hat \beta_1 \bar x, \hat \beta_1] \\
&= -\bar x Var[\hat \beta_1] \\
&= -\frac{\bar x \sigma^2}{S_{xx}}
\end{align}
$$


推导过程中用到了如下这个常用的结论，参见 [Blog](https://truenobility303.github.io/Hypothesis-Testing/)


$$
Cov[Y_i - \bar Y,\bar Y] = 0
$$


对于一元线性回归的结果也可以更加简单地表示，也是具有直观意义的结论，

$$
\begin{align}
\hat \beta_0 &= \bar y - \hat \beta_1 \bar x \\
\hat \beta_1 &= \frac{S_{xy}}{S_{xx}} \\
&=  \frac{\sum_{i=1}^N (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^N (x_i - \bar x)^2} \\
&= \frac{\sum_{i=1}^N x_i y_i - n \bar x \bar y}{\sum_{i=1}^N x_i^2 - n \bar x^2}
\end{align}
$$



在多元的情况下，我们知道$R^2$具有其直观的意义，表征了预测值和真实值之间的相关性，


$$
Corr^2[\hat Y, Y] = R^2
$$


而在一元的情况下，上式还有更加明显的含义，


$$
\begin{align}
R^2 &= Corr^2[\hat Y, Y] \\
&= Corr^2[\hat \beta X, Y] \\
&= Corr^2[X,Y] \\
&= \frac{S_{xy}^2}{S_{xx} S_{yy}}
\end{align}
$$


---

有时候也可以采用对数线性回归模型，例如用来解决异方差问题



$$
\log Y = X \beta + \epsilon, \epsilon \sim \mathcal{N}(0,\sigma^2)
$$



类似地可以得到参数得估计，$\hat \sigma^2, \hat \beta$ 



最终采用期望值作为最小均方预测可以得到，



$$
\begin{align}
E[Y] &= E[\exp(X \beta + \epsilon)] \\
&= E[\exp X \beta ] E[\exp \epsilon] \\
&=  \frac{\sigma^2}{2} \exp X \beta \\
\text{With } E[\exp \epsilon] &= \frac{\sigma^2}{2}, \text{By Generate Function}
\end{align}
$$



嵌入极大似然估计可以得到，

$$
E[\hat Y] = \frac{\hat \sigma^2}{2} \exp X \hat \beta
$$
但该结果并不再满足无偏估计



## Variable Selection Criteria 

模型的选择是一个普遍的问题。首先需要设定评价准则，之后是变量选择的问题，变量选择通常分为所有子集回归和基于贪心的方法。所有子集回归是枚举所有可能的组合，选择在设定的评价准则下最优的，但缺点是复杂度高。基于贪心的方法分为向前法、向后法和向前向后法，以使用单个变量的显著性F检验（实际上也为t检验）为例，向前法从空模型出发逐次添加对模型最为显著且超过给定置信度$\alpha$的变量直到没有变量是显著的，向后法从最大的模型出发逐次丢弃最不显著且低于给定置信度$\alpha$的变量直到不能再丢弃为止，而向前向后法每次添加一个变量，添加之后考虑是否新加入的变量可以使得已有的某个变量不显著，考虑是否将其丢弃，直到不能添加也不能丢弃为止。

---

下面介绍评价准则，最简单的评价准则是基于回归系数$R^2$或者调整后的回归系数$R_a^2$. 

由于$R^2$随着变量个数的增多而增大，因此对于$R^2$应该选择增益小的变量。而对于$R_a^2$，考虑了对过多的变量进行惩罚，选择$R_a^2$最大的模型即可。

### $\text{AIC}$

赤池信息准则（AIC）是一个著名的评价准则，其基于对数似然函数，并且对复杂度$p$进行惩罚，


$$
\begin{align}
AIC &= \max \log L_p - p \\
&= \max-\frac{n}{2} \log \hat \sigma^2 - \frac{\sum_i\Vert X_i \beta - Y_i \Vert_2^2}{2 \hat \sigma^2} -p\\
&= \max-\frac{n}{2} \log \hat \sigma^2 - \frac{n}{2} -p \\
&= \max -\frac{n}{2} \log \frac{1}{n} SSE - p \\
&=\max -\frac{n}{2} \log SSE - p
\end{align}
$$

### $C_p$

另一个评价准则称为$C_p$, 本质上是一种MSE的度量。

利用方差-偏差分解计算MSE，但此时并不一定是真模型下，因此偏差不一定为0，

$$
\begin{align}
MSE[\hat Y] &= tr Var[\hat Y] + \text{Bias}[\hat Y]^2 \\
&= tr(H \sigma^2) + (\mu - H \mu)^T (\mu - H \mu) \\
&= \sigma^2 p + \mu^T (I-H) \mu 
\end{align}
$$


而如果在真模型下，因为显然有$H \mu = \mu$, 因此在真模型下是无偏估计，而对于非真模型的情况，需要使用上面的公式比较MSE的大小，

同时，我们希望根据SSE进行估计，对其期望进行计算，首先由$EY = \mu,Var[Y] = \sigma^2$，因此根据方差和二阶矩的关系，


$$
\begin{align}
E[YY^T] &= Var[Y] + \mu \mu^T = \sigma^2 + \mu \mu^T \\
E[Y^T Y] &= tr Var[Y] + \mu^T \mu 
\end{align}
$$

利用上面的关系式可以直接得到，

$$
\begin{align}
E[SSE] &= E[Y^T(I-H) Y] \\
&= tr E[ (I-H)Y Y^T (I-H)] \\
&= tr Var[(I-H)Y] + \mu^T (I-H) \mu \\
&= \sigma^2 (n-p)  + \mu^T (I-H) \mu
\end{align}
$$


对比上下两式可以发现，


$$
\begin{align}
MSE - E[SSE] &= \sigma^2(2p-n) \\
\frac{MSE}{\sigma^2} &= \frac{E[SSE]}{\sigma^2} + (2p-n) \\
\end{align}
$$


由于$E[SSE],\sigma^2$都是未知的，采用$SSE,S$对其进行估计，由此得到$C_p$统计量，


$$
C_p = \frac{SSE}{S} + (2p-n)
$$
可以知道，$C_p$统计量是一种渐进无偏估计，也即，


$$
E[C_p] =\frac{E[SSE]}{\sigma^2} + (2p-n), n \rightarrow \infty
$$

可以根据$C_p$选择最小的，也即在选择$MSE$最小的模型，而且我们可以知道，在真实模型下，


$$
\begin{align}
E[C_p] &=\frac{E[SSE]}{\sigma^2} + (2p-n) = E[\chi^2(n-p)] +(2p-n) = p\\
\frac{MSE}{\sigma^2} &= p  + \frac{\mu^T (I-H) \mu}{\sigma^2} = p
\end{align}
$$


上述结论也从另一个角度验证了两者之间的关系，

### Relationship

另外一个有意思的结论是AIC准则和$C_p$准则之间的关系，当模型的方差$\sigma^2$已知的时候，两个选择选出来的模型是一致的，此时利用AIC的定义计算极大似然估计，

$$
\begin{align}
\text{AIC} &= \max L(\beta) - p \\
&= \max [-\frac{n}{2} \log \sigma^2  - \frac{(X\beta - y)^T(X\beta-y)}{2\sigma^2}] - p \\
&= \max [- \frac{(X\beta - y)^T(X\beta-y)}{2\sigma^2} - p]\\
&= -\frac{SSE}{2 \sigma^2} - p
\end{align}
$$


而在该假设下，选择$C_p$统计量最小的模型，与选择AIC准则最大的模型是一致的，


$$
\begin{align}
\min C_p &= \min [\frac{SSE}{\sigma^2} + (2p -n)] \\
&= \min[\frac{SSE}{\sigma^2} + 2p] \\
&=\min[\frac{SSE}{2\sigma^2} + p] \\
&= \max \text{AIC} 
\end{align}
$$

实际上，在渐进条件下，即便方差$\sigma^2$是未知的，利用Taylor展开，也可以证明上述两个准则是相同的，


$$
\begin{align}
\max \text{AIC} &= \max[-\frac{n}{2} \log \frac{1}{n} SSE - p] \\ 
&=\min [n \log \frac{SSE}{n} +2p] \\
&= \min [n(\log \sigma^2 + \log \frac{SSE}{n \sigma^2})+2p] \\
&\approx \min [n(\log \sigma^2 + \frac{SSE}{n \sigma^2}-1)+2p] \\
&= \min [ \frac{SSE}{2\sigma^2} + p] \\
&= \min C_p
\end{align}
$$


## Heteroscedasticity

本节我们考虑当回归模型不满足残差项方差相等情况，解决该问题的思路是带权最小二乘估计，对于异方差的回归可以对其进行归一化，


$$
\begin{align}
X_i &= Y_i \beta + \epsilon_i ,\epsilon \sim \mathcal{N}(0, diag(\sigma_i))\\
\frac{X_i}{\sigma_i} &= \frac{Y_i}{\sigma_i} \beta  + \frac{\epsilon_i}{\sigma_i} \\
\tilde X &= \tilde Y \beta + \tilde \epsilon, \tilde \epsilon \sim \mathcal{N}(0, I)
\end{align}
$$


对于上述模型求解最小方差无偏估计BLUE，利用之前的结论我们可以知道，


$$
\begin{align}
\hat \beta &= (\tilde X^T \tilde X)^{-1} \tilde X^T \tilde Y \\
&= (X^T WX)^{-1} X^T WY , W =diag(\frac{1}{\sigma_i^2}) 
\end{align}
$$



如果不信的话，简单验证上述结论即可，


$$
\begin{align}
Var[ \hat \beta + DY] &= Var[(X^T WX)^{-1} X^T WY +DY]\\
&= Var[\hat \beta]+Var[DY] + 2E[(X^T WX)^{-1} X^T WY Y^T D^T] \\
&= Var[\hat \beta]+Var[DY] + 2 (X^T WX)^{-1} X^T W E[Y Y^T] D^T \\
&= Var[\hat \beta]+Var[DY] + 2 (X^T WX)^{-1} X^T W E[Y Y^T] D^T \\
&= Var[\hat \beta]+Var[DY] + 2 (X^T WX)^{-1} X^T W (X W^{-1}X^T + X \beta \beta X^T) D^T  \\
&= Var[\hat \beta]+Var[DY] , \text{With } E[DY] = DX \beta = 0, DX = O
\end{align}
$$

类似地我们可以知道带权最小二乘估计，将获得该模型下的极大似然估计。

对于带权最小二乘估计的期望和方差进行计算，

$$
\begin{align}
E[\hat \beta] &= E[(X^T WX)^{-1} X^T WY] \\
&=E[(X^T WX)^{-1} X^T WX \beta ] \\
&= \beta \\
Cov[\hat \beta] &=Cov[(X^T WX)^{-1} X^T WY] \\
&=Cov[(X^T WX)^{-1}X^T W \epsilon] \\
&= (X^T WX)^{-1}X^T W E[\epsilon \epsilon^T] WX(X^T WX)^{-1} \\
&= (X^TWX)^{-1}
\end{align}
$$


而如果在异方差的模型上使用普通的最小二乘估计，虽然也是无偏估计，但方差并不是最小的，


$$
\begin{align}
E[ \hat \beta] &= \beta \\
Cov[\hat \beta] &= (X^TX)^{-1} X^T W^{-1}X (X^TX)^{-1}
\end{align}
$$

而对于方差未知的情况，我们就不能简单地使用带权最小二乘估计了，但实际上我们仍然可以随意指定一个$W_0$进行带权最小二乘估计，或者干脆指定$W=I$ 也即退化到最简单的普通最小二乘估计中，上述结果在$n \rightarrow \infty$ 的时候仍然是相合的，或者称为一致的

## Autocorrelation

用 [时间序列](https://truenobility303.github.io/ARIMA/) 的角度来看，我们如果将数据堪称带有时间序列特征的数据，例如假设残差项服从$AR(1)$ 模型，
$$
e_{t+1} = \rho e_t + \epsilon_t, \epsilon_t \sim \mathcal{N}(0,\sigma^2)
$$


经过简单的计算在该时间序列服从平稳性的时候，噪声项的期望和协方差应该满足下式，


$$
\begin{align}
E[e_t] &= 0\\
Cov[e_t] &= \frac{\sigma^2}{1- \rho^2}, \vert \rho \vert <1
\end{align}
$$


对于上述相关系数$\rho$的检验，最朴素的方法是采用矩估计，


$$
\hat \rho = \frac{\sum_{t=2}^N e_t e_{t-1}}{ \sum_{t=1}^N e_t^2}
$$


另一种检验被称为DW统计量 (Durbin–Watson Statistic)，其被定义为，


$$
\begin{align}
t_{DW} = \frac{\sum_{t=2}^N(e_t - e_{t-1})^2}{\sum_{t=1}^N e_t^2} \approx 2- 2\hat \rho \in [0,4]
\end{align}
$$


---

对于相关系数已知的情况，只需要进行差分变换，就可以化归为普通的线性回归的形式，


$$
\begin{align}
\rho Y_t &= \rho X_t \beta +\rho e_t \\
Y_{t+1} &= X_{t+1} \beta + e_{t+1} \\
Y_{t+1} - \rho X_t &= (X_{t+1}-\rho X_t) \beta + \epsilon_t, \epsilon_t \sim \mathcal{N}(0,\sigma^2) \\
\tilde Y_t &= \tilde X_t \beta +\epsilon_t,  \epsilon_t \sim \mathcal{N}(0,\sigma^2)
\end{align}
$$


而对于相关系数未知的情况，可以借助于 [EM算法](https://truenobility303.github.io/EM/) 中对于隐变量的处理手段，基于迭代的方法进行处理，

迭代法先使用普通的最小二乘估计得到残差的结果，之后对该残差的结果使用自回归，从而得到对于相关系数$\rho$的估计，


$$
e_{t+1} = \rho e_t +\epsilon_t, \epsilon_t \sim \mathcal{N}(0,\sigma^2)  
$$
得到了相关系数的估计量$\rho$之后，就化归为相关系数已知的情况，可以通过基于差分变换的方式进行回归，从而得到新的残差项，进一步又可以对新的残差使用自回归算法，周而复始地进行迭代。







