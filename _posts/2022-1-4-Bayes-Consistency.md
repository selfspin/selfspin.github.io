---
title: '代理性，最优性，可学性，一致性'
toc: true
excerpt_separator: <!--more-->
tags:
  - 统计机器学习
---



贝叶斯最优分类器是理论上最优的分类器，但由于其对应的0-1损失函数非凸不连续，难以直接进行优化，在机器学习中常用凸代理损失函数进行优化，本文研究使用代理损失函数是否能够一致地学习到贝叶斯最优分类器，称为算法的一致性。



<!--more-->



本节的大部分内容来自张潼2014年的论文:[Statistical behavior and consistency of classification methods based on convex risk minimization](https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/Statistical-behavior-and-consistency-of-classification-methods-based-on-convex/10.1214/aos/1079120130.full)  以及 《机器学习理论导引》.周志华等

 

## Surrogate Loss Function

机器学习中常用的代理损失函数有，


$$
\begin{align}
\phi(x) &= (x-1)^2 ,\text{Square Loss} \\
\phi(x) &= \max(0, 1- x), \text{Hinge Loss} \\
\phi(x) &= \log (1+ \exp(-x)), \text{Logistic Loss} \\
\phi(x) &= \exp(-x) , \text{Exponential Loss} \\
\end{align}
$$


如果绘制出上述函数的图像，可以发现其都是以下的贝叶斯损失，也即0-1损失的凸上界，


$$
R(x) = I(x \ne 1), \text{0-1 Loss}
$$


对于线性分类器，使用如上的损失可以得到经典的机器学习模型，

使用平方损失即可以得到线性回归模型，


$$
\min \frac{1}{N} \sum_{i=1}^N (1 - y_i (w^T x_i +b))^2
$$


使用合页损失（Hinge Loss）可以得到软间隔的支持向量机，


$$
\min \frac{1}{N}\sum_{i=1}^N\max(0, 1- y_i(w^Tx_i+b)) + \lambda \Vert w \Vert_2^2
$$


使用对率损失（Logistic Loss）可以得到对率回归模型，


$$
\min \frac{1}{N} \sum_{i=1}^N \log (1+ \exp(-y_i(w^T x_i + b)))
$$


使用指数损失，利用集成学习中的前向分布模型可以得到AdaBoost算法，


$$
\min \frac{1}{N} \sum_{i=1}^N \exp(-y_i \sum_{m=1}^M \alpha_m G_m(x))
$$


## Bayes Optimal



首先我们关心，在假设空间中，让代理损失函数取得最优值的假设，是否可以得到和贝叶斯最优分类器 $ f_{\star}$一样的分类结果。

我们假设给定一个数据 $x$,  其标签为1或0的概率为 $\eta(x)$, 则贝叶斯分类器选择概率较大的标签作为输出，可以得到最小的期望误差。 


$$
\begin{align}
R(f_{\star}) &= \min_f E_{(x,y)}[f(x) \ne y] \\
&= \min_f E_x [ \eta(x)I(f(x)=0)+(1-\eta(x))I(x=1))] \\
&= E_x[ \min(\eta(x), 1- \eta(x))] \\
\text{With } f_{\star}(x) &=I(\eta(x) >\frac{1}{2})
\end{align}
$$


为了证明的方便，我们后面假设 $\eta(x) \ne \frac{1}{2}$ .

该假设将使得证明中可以少处理该情况。而并不失一般性，对于该假设不成立的情况仍然可以类似地证明。

而最优化代理损失函数 $\phi(x) $ 也将得到一个对应的假设 $f_{\star}^\phi$ , 对应可以计算得到该假设最小的期望代理误差，


$$
\begin{align}
R_{\phi}(f_{\star}^\phi) &=\min_f E_{(x,y)} [\phi(f(x)y)] \\
&=\min_f E_{x} [\eta(x)\phi(f(x)) +(1-\eta(x)) \phi(-f(x)) ] \\
&= \min_f \eta(x)\phi(f(x)) +(1-\eta(x)) \phi(-f(x)), \forall x
\end{align}
$$


优化代理损失之后得到的结果通常不可能像优化0-1损失一样得到-1或1的直接标签，通常为一个-1到1之间的实数，可以理解为样本属于两类的概率。 实际使用中会将该概率的预测值与0对比并且选取概率大的作为预测标签的结果, 也即取符号运算。贝叶斯最优函数也即取符号运算后可以得到与贝叶斯最优分类器相同的结果，


$$
\text{sign}(f(x)) = I(\eta(x) > \frac{1}{2})
$$


可以证明的是，对上文提及的所有代理损失函数进行优化的最优值都满足贝叶斯最优函数，这些性质是代理损失的一致性的重要前提，



对于平方损失函数，


$$
\begin{align}
\text{Square Loss: } \phi(x) = (1-x)^2\\ 
\min R_{\phi}(f) &=\min_f \eta \phi(f) + (1-\eta) \phi(-f) , \text{Given } \eta = \eta(x) ,f =f(x), \forall x\\
&= \min_f \eta(f-1)^2 + (1-\eta)(f+1)^2 \\
\frac{\partial R_{\phi}}{\partial f} &= 2\eta(f-1) + 2(1-\eta)(f+1) = 0 \\
f_{\star}^\phi(x) &= 2 \eta(x) - 1 \\
\text{sign}(f_{\star}^\phi(x)) &= I(\eta(x) > \frac{1}{2})
\end{align}
$$


对于合页损失函数，由于其并不可导，需要对线性分段函数进行简单讨论即可得到答案，


$$
\begin{align}
\text{Hinge Loss: }\phi(x) &= \max(0,1-x) , \\ 
\min R_{\phi}(f) &=\min_f \eta \phi(f) + (1-\eta) \phi(-f) , \text{Given } \eta = \eta(x) ,f =f(x), \forall x\\
&= \min_f \eta \max(0,1-f) + (1-\eta)\max(0,1+f) \\
&= \min_{f = 0,1}  R_{\phi}(f) \\
f_{\star}^\phi(x) &= \text{sign} (2 \eta(x) - 1) \\
&= I(\eta(x) > \frac{1}{2}) \\
\end{align}
$$


对于指数损失函数，


$$
\begin{align}
\text{Exponential Loss: }\phi(x) &= \exp(-x) \\ 
\min R_{\phi}(f) &=\min_f \eta \phi(f) + (1-\eta) \phi(-f) , \text{Given } \eta = \eta(x) ,f =f(x), \forall x\\
&= \min_f \eta \exp(-f) + (1-\eta) \exp(f) \\
\frac{\partial R_{\phi}}{\partial f} &= -\eta \exp(-f) + (1-\eta)\exp(f) =0 \\
f_{\star}^{\phi} &=\frac{1}{2} \log \frac{\eta(x)}{1-\eta(x)} \\
\text{sign}(f_{\star}^\phi(x)) &= I(\eta(x) > \frac{1}{2}) \\
\end{align}
$$


对于对率损失函数，


$$
\begin{align}
\text{Logistic Loss: }\phi(x) &= \log (1+\exp(-x)) \\ 
\min R_{\phi}(f) &=\min_f \eta \phi(f) + (1-\eta) \phi(-f) , \text{Given } \eta = \eta(x) ,f =f(x), \forall x\\
&= \min_f \eta \log (1+\exp(-f)) + (1-\eta) \log (1+ \exp(f)) \\
\frac{\partial R_{\phi}}{\partial f} &=  \frac{-\eta \exp(-f)}{1+ \exp(-f)} + \frac{(1-\eta)\exp(f)}{1+\exp(f)} =0 \\
f_{\star}^{\phi} &= \log \frac{\eta(x)}{1-\eta(x)} \\
\text{sign}(f_{\star}^\phi(x)) &= I(\eta(x) > \frac{1}{2}) \\
\end{align}
$$


我们对得到的结果进行总结，可以得到对于代理损失函数来说，分别代入可以得到，理论上的最优的期望代理损失如下：


$$
\begin{align}
R_{\phi}(f_{\star}^\phi) &= 4E_x[\eta(x)(1-\eta(x))], \text{Square Loss} \\
R_{\phi}(f_{\star}^\phi) &= 2E_x[\min (\eta(x), 1-\eta(x))], \text{Hinge Loss} \\
R_{\phi}(f_{\star}^\phi) &= 2E_x[\sqrt{\eta(x)(1-\eta(x))}], \text{Exponential Loss} \\
R_{\phi}(f_{\star}^\phi) &= E_x[-\eta(x)\log \eta(x) - (1-\eta(x) \log(1-\eta(x)))], \text{Logistic Loss} \\
\end{align}
$$


由此我们证明了对于本文涉及的所有代理损失函数，都满足贝叶斯最优性，也即：

如果我们可以找到最优化代理损失函数的分类器，那么我们可以通过观察该分类器预测结果的符号来得到贝叶斯最优分类器。



## PAC Learnability

从贝叶斯最优性，我们可以发现学得的代理损失函数的最优分类器，至少在预测的符号上与贝叶斯分类器相一致。

但在实际训练的过程中，我们可能并不能求得代理损失函数的最优分类器，因为实际的训练过程基于有限样本计算，通常会选择经验风险最小化的假设作为输出，根据 [Hoeffding不等式](https://truenobility303.github.io/Probabilistic-Inequality/) , 我们可以知道经验风险最小化的假设可以趋近于期望风险最优的假设。



根据Hoeffding不等式，可以得到代理损失函数的最优假设的经验风险趋近于期望风险，

此时假设代理损失函数有界，也即 $\phi(x) \in [0,M]$



$$
\begin{align}
P(R_\phi(f_{\star}^\phi) \ge  \hat R_\phi(f_{\star}^\phi) + \epsilon ) &\le \exp(-\frac{2m \epsilon^2}{M}) \\
\hat R_\phi(f_{\star}^\phi) -  R_\phi(f_{\star}^\phi) &\le M\sqrt{\frac{1}{2m} \log \frac{1}{\delta}} , \text{With Prob.} 1 - \delta \\
\end{align}
$$




而对于训练集上的任意一个假设，类似地使用Hoeffding不等式，此时考虑假设空间有限的情况，而对于无限维假设空间的情况，基于VC维也可以给出类似的泛化误差界，可以参见 [VC维，复杂度，泛化界](https://truenobility303.github.io/VC-Dimension/) 



此时同样需要用到Hoeffding不等式，


$$
\begin{align}
P(R_\phi(f) > \hat R(f) + \epsilon, \exists f \in \mathcal{H}) &\le d P(R_\phi(f) > \hat R_\phi(f) + \epsilon) \\
&\le d \exp(-\frac{2m \epsilon^2}{M}) ,\text{By Hoeffding Inequality}\\
&= \delta 
\end{align}
$$
据此可以得到，


$$
\begin{align}
P(R_\phi(f) \le  \hat R_\phi(f) + M\sqrt{\frac{1}{2m} \log \frac{d}{\delta}}) &\ge  1- \delta ,\forall f \in \mathcal{H} \\
\hat R_\phi(f) -  R_\phi(f) &\le M\sqrt{\frac{1}{2m} \log \frac{d}{\delta}} , \text{With Prob.} 1 - \delta \\
\end{align}
$$



联合起来并且利用 $f$ 满足经验风险最小化可以得到PAC可学习性，


$$
\begin{align}
R_\phi(f) - R_\phi(f_{\star}^\phi) &\le \hat R_\phi(f) - R_\phi(f_{\star}^\phi) +  M\sqrt{\frac{1}{2m} \log \frac{2d}{\delta}} , \text{With Prob.} 1 - \frac{\delta}{2} \\
&\le \hat R_\phi(f_{\star}^\phi) - R_\phi(f_{\star}^\phi) +  M\sqrt{\frac{1}{2m} \log \frac{2d}{\delta}} , \text{With Prob.} 1 - \frac{\delta}{2} \\
&\le M\sqrt{\frac{1}{2m} \log \frac{2}{\delta}} + M\sqrt{\frac{1}{2m} \log \frac{2d}{\delta}} , \text{With Prob.} 1 - \delta \\
\end{align}
$$


该部分的证明实际上与 [稳定性，泛化性，可学性，正则化](https://truenobility303.github.io/Stability/) 中关于稳定性与PAC可学习性的证明基本一致。

解读根据上面的证明过程得到的PAC可学习性的含义，我们可以发现：

使用经验风险最小化得到的分类器 $f$ 可以大概率逼近对于代理损失函数的最优分类器，且收敛率由概率不等式给出。



## Bayes Consistency

在上一节的可学习性分析中，我们知道使用经验风险最小化，可以学得到对于代理损失函数的最优分类器的很好的近似。

但由于我们关心的目标为0-1损失函数，我们希望对于代理损失函数的最优分类器的近似，也可以同时成为0-1损失函数的近似，这样的性质被称为贝叶斯一致性。下面我们将证明，对于上述的所有代理损失函数，贝叶斯一致性都是成立的，也就是说虽然我们在优化代理损失函数，但实际上我们可以学得到贝叶斯最优分类器很好的近似。

上述结论是非常神奇的，因为我们在前面已经求得了对于代理损失函数的最优分类器的形式，但这仅仅是理论结果，实际上我们并不一定能求得该分类器。但贝叶斯一致性告诉我们，我们只需要求得对于代理损失函数的最优分类器的一个近似就足够了，此时我们等价于求解了贝叶斯最优分类器的近似。



首先定义与代理损失函数相关的Q函数和P函数，据此可以简化期望代理损失和期望0-1损失的表达式，


$$
\begin{align}
\text{Define }
Q(f) &= \eta(x) \phi(f(x)) + (1-\eta(x)) \phi(f(x) ) \\
P(f) &= \eta(x) I(f(x)=0) + (1-\eta(x)) I(f(x)=1)\\
\text{Then } 
R_\phi(f) &= E_x[Q(f)] \\
R(f) &= E_x[P(f)]
\end{align}
$$


借助Q函数的定义，我们给出下面序贝叶斯一致性的普遍定理，后面我们将看到上述的所有代理损失函数，都可看作是该定理的特例，


$$
\begin{align}
\text{Gievn }  \text{sign}(f_{\star}^\phi(x)) &= I(\eta(x) > \frac{1}{2}) \\ 
\vert \eta(x) - \frac{1}{2} \vert^s &\le c^s(Q(0)- Q(f_{\star}^\phi)), \exists c>0, \exists s\ge 1 \\
\text{Then } R_\phi(f) \rightarrow R_\phi(f_{\star}^\phi) &\Rightarrow R(f) \rightarrow R(f_{\star})
\end{align}
$$


在给出定理的证明之前，我们先解读一下这个看上去不大好理解的定理，

首先定理要求最优的代理分类器满足贝叶斯最优性，也即 $\text{sign}(f_{\star}^\phi(x)) = I(\eta(x) > \frac{1}{2})$ 

其次定理要求最优代理分类器的损失 $Q(f_{\star}^\phi)$ 相较于随机猜测的损失 $Q(0)$ 应该至少有如下关于条件概率 $\eta(x)$ 相关的提升， 

也即表达式的含义：  $\vert \eta(x) - \frac{1}{2} \vert^s \le c^s(Q(0)- Q(f_{\star}^\phi)), \exists c>0, \exists s\ge 1 \\$ 

最终可以推出，如果我们可以得到一个逼近最优代理分类器的函数序列，则该函数序列也可以逼近贝叶斯最优分类器，

---

由于定义的代理损失函数均为凸函数，此时与数据集上的损失相关的Q函数也为凸函数，可以直接验证定义即可，


$$
\begin{align}
Q(\alpha f + (1- \alpha g)) &= \eta \phi(\alpha f + (1-\alpha) g) + (1-\eta) \phi(-\alpha f -(1-\alpha g) ) \\
&\le \eta \alpha \phi(f) +\eta (1-\alpha) \phi(g) + (1-\eta) \alpha \phi(-f) + (1-\eta)(1-\alpha)\phi(-g)\\
&=\alpha Q(f) +(1-\alpha) Q(g) \\ 
\text{Let } \eta &= \eta(x), f = f(x), g = g(x) 
\end{align}
$$


学习到的近似分类器 $f$ 的缺陷是其不满足贝叶斯最优性，也即其预测结果的符号可能与贝叶斯最优分类器的符号不相同，这将造成近似分类器与贝叶斯最优分类器表现上的差异，因此我们考虑该差异即可，

坏消息是该差异使得分类器表现下降，好消息是该差异告诉我们此时近似分类器和贝叶斯最优分类器异号，也即位与 $Q(0)$ 的两侧，可以根据 $Q$ 为凸函数的性质利用两个分类器的结果估计 $Q(0)$ 也即随机猜测的风险，下面我们给出完整证明，


$$
\begin{align}
R(f) - R(f_{\star}) &= E_x [P(f) - P(f_{\star})] \\
&=2E_{\text{sign} (f) \ne \text{sign}(f_{\star})} \vert \eta(x) - \frac{1}{2} \vert ,\text{By Simple Calculation} \\
& \le 2 E_{\text{sign} (f) \ne \text{sign}(f_{\star})} [\vert \eta(x) - \frac{1}{2} \vert^s]^{\frac{1}{s}} ,\text{Jesson Inequality} \\
&\le 2 c E_{\text{sign} (f) \ne \text{sign}(f_{\star})} [Q(0) - Q(f_{\star}^\phi)]^{\frac{1}{s}} \\
&\le 2 c E_{\text{sign} (f) \ne \text{sign}(f_{\star})} [\max [Q(f),Q(f_{\star}^\phi)]-Q(f_{\star}^\phi)]^{\frac{1}{s}}, \text{By Convexity of } Q   \\
&= 2 c E_{\text{sign} (f) \ne \text{sign}(f_{\star})} [Q(f) - Q(f_{\star}^\phi)]^{\frac{1}{s}} \\
&\le 2cE_x [Q(f) - Q(f_{\star}^\phi)]^{\frac{1}{s}} \\
\end{align}
$$


最终的结果蕴含了定理的结论中的收敛性，从而完成了整个证明。



---

最终我们考虑定理的应用，回归到最开出给出的几种常见的代理损失函数，利用上述定理，证明所有的上述代理损失函数都满足贝叶斯一致性，我们将看到尽管定理形式略显复杂，但应用却十分方便，只需要计算 $Q(0) - Q(f_{\star}^\phi)$ 的值即可



对于平方损失函数，


$$
\begin{align}
\text{Square Loss: } \phi(x) &= (1-x)^2\\ 
Q(0) &= 1, Q(f_{\star}^\phi) = 4 \eta (1- \eta) \\
Q(0) - Q(f_{\star}^\phi) &= 1 - 4 \eta (1- \eta) \\
&=4(\eta - \frac{1}{2})^2 \\
&= c^s \vert \eta - \frac{1}{2} \vert^s, \text{Let } c = 2, s = 2 
\end{align}
$$


对于合页损失函数，


$$
\begin{align}
\text{Square Loss: } \phi(x) &= \max(0,1-x)\\ 
Q(0) &= 1, Q(f_{\star}^\phi) = 2 \min (\eta, 1- \eta) \\
Q(0) - Q(f_{\star}^\phi) &= 1 - 2 \min (\eta, 1- \eta) \\
&=2 \vert \eta - \frac{1}{2} \vert \\
&= c^s \vert \eta - \frac{1}{2} \vert^s, \text{Let } c=2,s=1
\end{align}
$$


对于指数损失函数，


$$
\begin{align}
\text{Square Loss: } \phi(x) &= \exp(-x)\\ 
Q(0) &= 1, Q(f_{\star}^\phi) = 2 \sqrt{\eta(1-\eta)} \\
Q(0) - Q(f_{\star}^\phi) &= 1-2 \sqrt{\eta(1-\eta)} \\
&=  \frac{2- 4 \sqrt{\eta(1-\eta)}}{2}  \\
&=\frac{1- 4 \sqrt{\eta(1-\eta)} + 4\eta(1-\eta) - 4\eta(1-\eta) +1}{2} \\
&=\frac{(2\sqrt{\eta (1-\eta)} - 1)^2 - 4\eta(1-\eta) +1}{2} \\
&\le \frac{ 1- 4\eta(1-\eta) }{2} \\
&= 2(\eta - \frac{1}{2})^2 \\
&= c^s \vert \eta - \frac{1}{2} \vert^s, \text{Let } c=\sqrt 2,s=2
\end{align}
$$


对于对率损失函数，使用Taylor展开也可以证明，


$$
\begin{align}
\text{Logistic Loss: }\phi(x) &= \log(1+ \exp(-x)) \\
Q(0) &=1 , Q(f_{\star}^\phi) =-\eta \log \eta -(1-\eta) \log (1-\eta)  \\
\text{Let } g(\eta) &= Q(f_{\star}^\phi) =-\eta \log \eta -(1-\eta) \log (1-\eta) \\
\nabla g &= -\log \eta + \log (1-\eta)\\ 
\nabla^2 g &= -\frac{1}{\eta(1-\eta)} \\
Q(0) - Q(f_{\star}^\phi)  &= g(\frac{1}{2}) - g(\eta) \\
&= g(\frac{1}{2}) - (g(\frac{1}{2})  - \frac{1}{\eta' (1-\eta')} (\eta - \frac{1}{2})^2) ,\exists \eta' \in (0,\eta) \\
&= \frac{1}{\eta'(1-\eta')} (\eta - \frac{1}{2})^2 \\
&\le 2 (\eta - \frac{1}{2})^2 \\
&= c^s \vert \eta - \frac{1}{2} \vert^s, \text{Let } c = \sqrt 2, s=2
\end{align}
$$


据此我们成功地证明了上述代理损失函数都满足贝叶斯一致性，这为SVM、Logistic Regression、AdaBoost等方法建立了理论基础。




$$
\begin{align}

\end{align}
$$




