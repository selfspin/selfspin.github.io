---
title: '稳定性，泛化性，可学性，正则化'
toc: true
excerpt_separatore: <!--more-->
tags:
  - 统计机器学习
---



稳定性衡量机器学习算法对于训练集的扰动的稳健性，本文主要针对于替换稳定性，从替换稳定性的定义出发，推导稳定性和泛化性以及PAC可学性的关系，并且基于支持向量机、岭回归等经典模型推导稳定性和正则化的关系。



<!--more-->





在 [VC维，复杂度，泛化界](https://truenobility303.github.io/VC-Dimension/) 中介绍了假设空间复杂度以及于其相关的泛化误差上界，但在实际机器学习的过程中我们关心具体的算法的泛化性，由此首先需要算法对于不同的数据集结果具有稳定性，因此引出稳定性的概念。



## Replacement Stability

讨论替换稳定性，对于训练集 $D$, 对其施加小的扰动，也即将训练集中的某个样本替换为另一个样本，从而得到训练集 $D'$

一个具有替换稳定性的算法希望对于扰动前后的两个数据集，对于任意样本 $z$, 算法的结果关于该样本的损失变化很小，假设算法在训练集上训练后得到的结果可以用权重 $w$ 表示，


$$
\vert R(w_D,z) - R(w_{D'}, z ) \vert < \beta, \exists \beta 
$$


满足上述定义的机器学习算法具有替换样本 $\beta$ - 均匀稳定性。



### Stability and Generalization

本节讨论算法的稳定性以及泛化性之间的关系。



其中需要用到的重要工具为McDiarmid不等式，其可以看作Hoeffding不等式的更普遍情况，


$$
\begin{align}
\text{If } f(x_1,...,x_i,...,x_m) - f(x_1,...,x_i',...,x_m) \vert &\le c_i ,\forall x_i \in X_i, X_i  \text{ Independent} \\
\text{Then } P(f(X_1,...,X_m)- Ef(X_1,...,X_m) \ge \epsilon ) &\le \exp(-\frac{2 \epsilon^2}{\sum_{i=1}^m c_i^2}) \\
P(f(X_1,...,X_m)- Ef(X_1,...,X_m) \le -\epsilon ) &\le \exp(-\frac{2 \epsilon^2}{\sum_{i=1}^m c_i^2}) \\
\end{align}
$$


可以看到该不等式的条件与替换稳定性密切相关，因此该不等式将在证明中扮演着重要作用。

为了计算泛化误差界，定义与泛化误差直接相关的函数并对其直接进行分析，且假设损失函数是有界的，


$$
\Phi(D) = R(w_D) - \hat R(w_D) = \Phi(z_1,...,z_m)
$$


考虑替换训练集中某个样本后得到的扰动后的训练集 $D'$, 其将原先的样本 $z_i$ 替换为新样本 $z_i'$ , 


$$
\begin{align}
\vert \Phi(D) - \Phi(D') \vert &= \vert (R(w_D) - \hat R(w_D)) -(R(w_D') - \hat R(w_D')) \vert \\
&\le  \vert \hat R(w_D) - \hat R(w_D') \vert+\vert R(w_D) -  R(w_D') \vert  \\
&\le \frac{\vert R(w_D,z_i) - R(w_D',z_i') \vert}{m} + \sum_{j \ne i}\frac{\vert R(w_D,z_j) - R(w_D',z_j)  \vert}{m} + \vert R(w_D) -  R(w_D') \vert  \\
&= \frac{\vert R(w_D,z_i) - R(w_D',z_i') \vert}{m} + \sum_{j \ne i}\frac{\vert R(w_D,z_j) - R(w_D',z_j)  \vert}{m} + \vert E_z[R(w_D,z) -  R(w_D',z)] \vert  \\ 
&\le \frac{M}{m} + \frac{(M-1) \beta}{M} +  \beta \\
&\le \frac{M}{m} + 2 \beta
\end{align}
$$


因此该函数满足McDiarmid不等式的条件，下面求解函数 $\Phi$ 的期望的上界估计，


$$
\begin{align}
E_D[\Phi(D)] &= E_D[R(w_D) - \hat R(w_D)] \\
&= E_{D,z'} [R(w_D,z') - R(w_D', z')] \\
&\le \beta
\end{align}
$$


上式利用到扰动后的数据集 $D'$ 包含样本 $z'$ 的限制，将经验风险函数的期望表达为上式，再使用稳定性的条件，最终可以应用McDiarmid不等式得到，


$$
\begin{align}
P(\Phi(D) \ge \beta +\epsilon) &\le P(\Phi(D) \ge E [\Phi(D)]  +\epsilon) \\
&\le \exp(-\frac{2\epsilon^2}{m(\frac{M}{m}+ 2\beta)^2}) \\
&= \exp(-\frac{2m\epsilon^2}{(M+ 2m \beta)^2}) \\
&= \delta
\end{align}
$$
据此可以得到有下列高概率成立的不等式，


$$
\begin{align}
R(w_D) <  \hat R(w_D) + \beta + (2m \beta+M) \sqrt{\frac{1}{2m} \log \frac{1}{\delta}} ,\text{With Prob.} 1- \delta
\end{align}
$$


后面我们将看到，很多经典的机器学习算法来说，满足 $m \beta = C, \exists C$ , 因此基于稳定给给出的泛化误差界的收敛率为 $O(\frac{1}{\sqrt m})$ , 该结果与有限维假设空间的泛化误差收敛率相同，但却是一个维度无关的界，因此该结论同样适用于无限维假设空间。也即如果算法可以满足上述稳定性，根据稳定性可以分析得到更紧的泛化性估计。 



### Stability and Learnability

PCA可学习性（Probably Approximately Correct Learnability）要求学习算法可以在高概率的前提下，学习到关于假设空间中的最优函数的近似，而最优的函数定义为泛化误差最小的函数，也即期望意义下的最优函数，将其记作为 $w_{\star}$. 



根据Hoeffding不等式，可以得到，


$$
\begin{align}
P(R(w_{\star}) \ge  \hat R(w_{\star}) + \epsilon ) &\le \exp(-\frac{2m \epsilon^2}{M}) \\
\hat R(w_{\star}) -  R(w_{\star}) &\le M\sqrt{\frac{1}{2m} \log \frac{1}{\delta}} , \text{With Prob.} 1 - \delta \\
\end{align}
$$


在实际学习过程中，通常基于经验风险最小化，此时可以得到其可学性，


$$
\begin{align}
R(w_D) - R(w_{\star}) & \le \hat R(w_{D})- R(w_{\star})  + \beta + (2m \beta+M) \sqrt{\frac{1}{2m} \log \frac{2}{\delta}} ,\text{With Prob.} 1- \frac{\delta}{2}\\
& \le \hat R(w_{\star})- R(w_{\star})  + \beta + (2m \beta+2M) \sqrt{\frac{1}{2m} \log \frac{2}{\delta}}  \\
&\le \beta + (2m \beta+2M) \sqrt{\frac{1}{2m} \log \frac{2}{\delta}} ,\text{With Prob.} 1- \delta
\end{align}
$$


## Stability and Regularization

正则化是机器学习中的重要技巧，本节将基于经典的机器学习模型，利用理论推导给出正则化与稳定性的关系，从理论的层面说明正则化为何能否减小泛化误差，从而避免过拟合的现象发生。



### SVM

考虑带正则项的软间隔支持向量机（Support Vector Machine，SVM），也即优化目标函数为，


$$
\min F_D(w) := \frac{1}{m} \sum_{i=1}^m \max(0,1- y_i w^T x_i) + \lambda \Vert w \Vert_2^2
$$


根据强凸性以及最优解的一阶梯度为0的性质可以得到，


$$
\begin{align}
F_D(w_D') &\ge F_D(w_D) +  \lambda \Vert w_D' - w_D \Vert_2^2 \\
F_D'(w_D) &\ge F_D'(w_D') +  \lambda \Vert w_D' - w_D\Vert_2^2 \\
\end{align}
$$


并且假设 $\Vert x_i \Vert \le r$ ,此时可以得到，


$$
\begin{align}
\vert \max(0, 1 - y w_D^T x) - \max(0,1-yw_D'^Tx) \vert &\le \vert yw_D^T x- y w_D'^T x \vert \\
&\le r \Vert w_D - w_D' \Vert
\end{align}
$$


因此可以有，


$$
\begin{align}
\Vert w_D' - w_D \Vert_2^2  &\le \frac{F_D(w_D') - F_D(w_D)+F_D'(w_D) - F_D'(w_D')}{2 \lambda} \\
&\le \frac{\vert F_D(w_D') - F_D(w_D)\vert + \vert F_D'(w_D) - F_D'(w_D') \vert }{2 \lambda} \\
&\le \frac{r}{m \lambda} \Vert w_D' - w_D \Vert  \\
\Vert w_D' - w_D \Vert &\le \frac{r}{m \lambda}
\end{align}
$$


最终反代入前式则可以证明正则化的SVM满足替换稳定性，


$$
\begin{align}
\vert \max(0, 1 - y w_D^T x) - \max(0,1-yw_D'^Tx) \vert &\le \vert yw_D^T x- y w_D'^T x \vert \\
&\le r \Vert w_D - w_D' \Vert \\
&\le \frac{r^2}{m \lambda}
\end{align}
$$


利用稳定性与泛化性的关系可以得到其泛化界的收敛率为 $O(\frac{1}{\sqrt{m}})$ 


$$
\begin{align}R(w_D) &<  \hat R(w_D) + \frac{r^2}{m \lambda} + (\frac{2r^2}{ \lambda}+M) \sqrt{\frac{1}{2m} \log \frac{1}{\delta}} ,\text{With Prob.} 1- \delta \\
\end{align}
$$


### Ridge Regression

岭回归（Ridge Regression）,可以参见 [岭回归](https://truenobility303.github.io/Rregression-Second/) ，是正则化后的线性回归模型，

为了分析岭回归的泛化性，我们假设自变量有界 $\Vert x \Vert \le r$  以及损失函数也有界 $R(w_D,x) \in [0,M]$ 的前提，此时优化的目标函数为，


$$
\min F_D(w) := \frac{1}{m} \sum_{i=1}^m (y_i - w^T x_i)^2 + \lambda \Vert w \Vert_2^2
$$
根据有界性的前提假设可以得到，


$$
\begin{align}
\vert (y - w_D^T x)^2 - (y - w_D'^T x)^2 \vert &\le \vert w_D^Tx - w_D'x \vert \vert (y-w_D^Tx)+(y - w_D'^Tx) \vert \\
&\le 2 r \sqrt M \Vert w_D - w_D' \Vert 
\end{align}
$$


后面的证明过程完全类似，




$$
\begin{align}
\Vert w_D' - w_D \Vert_2^2  &\le \frac{F_D(w_D') - F_D(w_D)+F_D'(w_D) - F_D'(w_D')}{2 \lambda} \\
&\le \frac{\vert F_D(w_D') - F_D(w_D)\vert + \vert F_D'(w_D) - F_D'(w_D') \vert }{2 \lambda} \\
&\le \frac{2r \sqrt M}{m \lambda} \Vert w_D' - w_D \Vert  \\
\Vert w_D' - w_D \Vert &\le \frac{2r \sqrt M}{m \lambda}
\end{align}
$$


最终反代入前式则可以证明岭回归满足替换稳定性，


$$
\begin{align}
\vert (y - w_D^T x)^2 - (y - w_D'^T x)^2 \vert &\le \vert w_D^Tx - w_D'x \vert \vert (y-w_D^Tx)+(y - w_D'^Tx) \vert \\
&\le 2 r \sqrt M \Vert w_D - w_D' \Vert \\
&\le  \frac{4Mr^2}{m \lambda} \Vert w_D - w_D' \Vert \\
\end{align}
$$


利用稳定性与泛化性的关系可以得到其泛化界的收敛率也为 $O(\frac{1}{\sqrt{m}})$ 


$$
\begin{align}R(w_D) &<  \hat R(w_D) + \frac{4 Mr^2 }{m \lambda} + (\frac{8Mr^2}{ \lambda}+M) \sqrt{\frac{1}{2m} \log \frac{1}{\delta}} ,\text{With Prob.} 1- \delta \\
\end{align}
$$

---

参数 $\lambda$ 作为正则化因子，其惩罚力度越大，可以发现泛化界将越紧。

上面的证明过程说明了正则项实际上使得算法学习更为稳定，对于扰动更加稳健。

相似的结论对于很多其他机器学习模型也是成立的，证明过程也基本完全类似，包括对率回归模型（Logistic Regression）、支持向量回归（Support Vector Regression）等。