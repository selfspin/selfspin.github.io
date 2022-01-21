---
title: 'L-SVRG and L-Katyusha'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 随机优化
---

论文阅读笔记：[Don’t Jump Through Hoops and Remove Those Loops: SVRG and Katyusha are Better Without the Outer Loop](http://proceedings.mlr.press/v117/kovalev20a)

<!--more-->

随机方差缩减梯度方法（SVRG）及其加速版本（Katyusha）的简易版变种，分别称为L-SVRG和L-Latyusha

## Problem Set-up

关注于机器学习中常见的经验风险最小化问题，


$$
\min f(x) = \frac{1}{n} \sum_{i=1}^n f_i(x)
$$


并且假设优化的每一个函数 $f_i(x)$ 都满足L-光滑和 $\mu$-强凸性质，也即，


$$
\begin{align}
f_i(y) &\le f_i(x) + \nabla f_i(x)^T(y-x) + \frac{L}{2} \Vert y - x \Vert_2^2 \\
f_i(y) &\ge f_i(x) + \nabla f_i(x)^T(y-x) + \frac{\mu}{2} \Vert y - x \Vert_2^2 \\
\end{align}
$$


## L-SVRG



算法改进了原本的 [SVRG算法](https://truenobility303.github.io/SVRG/) ，原本的SVRG需要每隔 $m$ 轮维护一次样本的全梯度，而L-SVRG采用每次迭代以一定的概率进行该维护，从而避免了SVRG维护梯度的外层循环。


$$
\begin{align}
g_k &= \nabla f_i(x_k) - \nabla f_i (w_k) + \nabla f(w_k), i \sim \mathcal{U}(1..n) \\
x_{k+1} &= x_k - \eta g_k \\
w_{k+1} &= x_k \text{ With Prob. p}  \\
&=w_k \text{With Prob. } 1-p
\end{align}
$$


下面我们证明L-SVRG方法的收敛率，需要用到如下的Lyapunov函数，


$$
\begin{align} 
\Phi_k &= \Vert x_k - x_{\star} \Vert_2^2 + \mathcal{D_k} \\
&= \Vert x_k - x_{\star} \Vert_2^2 + \frac{4 \eta^2}{p} E[ \Vert \nabla f_i(w_k) - \nabla f_i(x_{\star}) \Vert_2^2] \\ 
&=\Vert x_k - x_{\star} \Vert_2^2 + \frac{4 \eta^2}{pn} \sum_{i=1}^n \Vert \nabla f_i(w_k) - \nabla f_i(x_{\star}) \Vert_2^2 \\ 
\end{align}
$$


随机优化中随机选取样本计算梯度进行下降，对所有的样本取期望，每次迭代中的随机性仅仅在 $g_k$ 上体现，


$$
\begin{align}
E[\Vert x_{k+1} - x_{\star} \Vert_2^2] &= E[\Vert x_k - \eta g_k  - x_{\star} \Vert_2^2] \\
&=\Vert x_k - x_{\star} \Vert_2^2  - 2 \eta E[g_k^T(x_k - x_{\star})] + \eta^2E[\Vert g_k \Vert_2^2]\\
&= \Vert x_k - x_{\star} \Vert_2^2  - 2 \eta  \nabla f(x_k)^T(x_k - x_{\star}) + \eta^2E[\Vert g_k \Vert_2^2]\\
&= \Vert x_k - x_{\star} \Vert_2^2  + 2 \eta  \nabla f(x_k)^T(x_{\star} - x_{k}) + \eta^2E[\Vert g_k \Vert_2^2]\\
&\le  \Vert x_k - x_{\star} \Vert_2^2  + 2 \eta (f(x_{\star})- f(x_k) - \frac{\mu}{2} \Vert x_{\star} - x_k \Vert_2^2 ) + \eta^2E[\Vert g_k \Vert_2^2], \text{By Strongly Convexity}\\ 
&= (1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 + 2 \eta (f(x_{\star})-f(x_k)) + \eta^2E[\Vert g_k \Vert_2^2] \\
&= (1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 - 2 \eta (f(x_{k})-f(x_{\star})) + \eta^2E[\Vert g_k \Vert_2^2] \\
\end{align}
$$


使用 $\mathcal{D_k}$ 限制 $g_k$ 的范数的界，


$$
\begin{align}
E [\Vert g_k \Vert_2^2] &= E[ \Vert \nabla f_i(x_k) - \nabla f_i (w_k) + \nabla f(w_k) \Vert_2^2] \\
&=E[ \Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) +\nabla f_i(x_{\star}) -\nabla f_i (w_k) + \nabla f(w_k) \Vert_2^2] \\
&\le 2E [\Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] +2 E[\Vert \nabla f_i(x_{\star}) -\nabla f_i (w_k) + \nabla f(w_k) \Vert_2^2] \\
&= 2E [\Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] +2 Var[\nabla f_i(x_{\star}) -\nabla f_i (w_k)] \\
&\le 2E [\Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] + 2 E[\Vert \nabla f_i (w_k) - \nabla f_i(x_{\star}) \Vert_2^2] \\
\end{align}
$$


使用SVRG中的性质可以得到，


$$
\begin{align}
\text{Let } g_i(x) &= f_i(x)- f_i(x_{\star}) -\nabla f_i(x_{\star})^T(x- x_{\star}) \\
0 = g_i(x_{\star}) &\le \min_{\eta} [g_i(x- \eta \nabla g_i(x))] \\
&\le \min_{\eta} [g_i(x) - \eta \Vert g_i(x) \Vert_2^2 + \frac{1}{2} L \eta^2 \Vert \nabla g_i(x) \Vert_2^2] \\
&=g_i(x) - \frac{1}{2L} \Vert \nabla g_i(x) \Vert_2^2 \\
E [\Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] &= E[\Vert \nabla g_i(x_k) \Vert_2^2] \\
&\le 2L E[g_i(x_k)- g_i(x_{\star})] \\
&= 2LE[f_i(x_k) - f_i(x_{\star} ) - \nabla f_i(x_{\star})^T(x_k - x_{\star})] \\
&= 2LE[f_i(x_k) - f_i(x_{\star} )] \\
&=2L [f(x_k) - f(x_{\star}) ]
\end{align}
$$


代入就可以得到了，


$$
\begin{align}
E [\Vert g_k \Vert_2^2] &\le 2E [\Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] + 2 E[\Vert \nabla f_i (w_k) - \nabla f_i(x_{\star}) \Vert_2^2] \\ 
&\le 4L(f(x_k) - f(x_{\star})) + \frac{p}{2 \eta^2} \mathcal{D_k} \\
\end{align}
$$


进而如下控制 $\mathcal{D_k}$ 的界， 


$$
\begin{align}
E[\mathcal{D_{k+1}}] &= (1-p)\mathcal{D_k} + p \frac{4 \eta^2}{p} E[ \Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] \\ 
&=(1-p)\mathcal{D_k} + 4 \eta^2 E[ \Vert \nabla f_i(x_k) - \nabla f_i(x_{\star}) \Vert_2^2] \\ 
&\le (1-p)\mathcal{D_k} + 8 \eta^2L (f(x_k) - f(x_{\star})) \\ 
\end{align}
$$

最终可以得到Lyapunov函数的收敛性，



$$
\begin{align}
E[\Phi_{k+1}] &= E[\Vert x_{k+1} - x_{\star} \Vert_2^2 + \mathcal{D_{k+1}}] \\
& \le (1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 - 2 \eta (f(x_{k})-f(x_{\star})) + E[\eta^2\Vert g_k \Vert_2^2 + \mathcal{D_{k+1}} ] \\
&\le (1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 - 2 \eta (f(x_{k})-f(x_{\star})) + 4\eta^2 L(f(x_k) - f(x_{\star})) + \frac{p}{2} \mathcal{D_k} + E[\mathcal{D_{k+1}} ] \\
&\le (1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 - 2 \eta (f(x_{k})-f(x_{\star})) + 4\eta^2 L(f(x_k) - f(x_{\star})) + \frac{p}{2} \mathcal{D_k} + (1-p)\mathcal{D_k} + 8 \eta^2L (f(x_k) - f(x_{\star}) \\
&=(1-  \eta \mu)\Vert x_k - x_{\star} \Vert_2^2 +(12\eta^2L - 2 \eta)(f(x_{k})-f(x_{\star})) + (1-\frac{p}{2} )\mathcal{D_{k}} \\
&=(1-  \frac{\mu}{6L} )\Vert x_k - x_{\star} \Vert_2^2  + (1-\frac{p}{2} )\mathcal{D_{k}} ,\text{Set } \eta = \frac{1}{6L}\\ 
&\le \max(1 - \frac{\mu}{6L} , 1- \frac{p}{2}) \Phi_k
\end{align}
$$



可以解得为了达到 $\epsilon$-最优解，需要的迭代次数为，


$$
\begin{align}
E[\Phi_k] &\le \max(1 - \frac{\mu}{6L} , 1- \frac{p}{2})^k \Phi_0 \le \epsilon \Phi_0 \\
k &\ge \max(\frac{6L}{\mu},\frac{2}{p}) \log \frac{1}{\epsilon}
\end{align}
$$


考虑计算梯度的迭代次数，可以得到L-SVRG算法的期望复杂度，


$$
\begin{align}
k' &\ge (1+ pn)  (\frac{6L}{\mu} +\frac{2}{p}) \log \frac{1}{\epsilon} \\
&= (12 \kappa +2n) \log \frac{1}{\epsilon} ,\text{Set } p = \frac{1}{n}, \text{Let } \kappa = \frac{L}{\mu}
\end{align}
$$


也即需要的梯度计算次数为 $O((\kappa + n) \log \frac{1}{\epsilon} ) $, 其中 $\kappa = \frac{L}{\mu}$ 为函数的条件数，刻画了光滑系数 $L$ 和强凸系数 $\mu$ 之间的差距。



## L-Katyusha

L-Katyusha是L-SVRG的加速版本，来源于Katyusha加速，算法的改进思路相同，但证明过程稍显复杂，算法迭代如下，


$$
\begin{align}
x_k &= \theta_1 z_k + \theta_2 w_k + (1-\theta_1 - \theta_2) y_k \\
g_k &= \nabla f_i(x_k) - \nabla f_i (w_k) + \nabla f(w_k), i \sim \mathcal{U}(1..n) \\
z_{k+1} &=\frac{1}{1+ \eta \sigma}(\eta \sigma x_k + z_k - \frac{\eta}{L} g_k), \text{With } \sigma = \frac{1}{\kappa} \\ 
y_{k+1} &=x_k + \theta_1 (z_{k+1} - z_k) \\
w_{k+1} &= y_k \text{ With Prob. p}  \\
&=w_k \text{With Prob. } 1-p
\end{align}
$$

L-Katyusha的收敛性依赖于如下的Lyapunov函数，



$$
\begin{align}
\Phi_k &= \mathcal{Z_k} + \mathcal{Y_k} + \mathcal{W_k} \\
&= \frac{L(1+ \eta \sigma)}{2 \eta } \Vert z_k - x_{\star} \Vert_2^2 + \frac{1}{\theta_1} (f(y_k)-f(x_{\star})) + \frac{\theta_2 (1+ \theta_1)}{p \theta_1}(f(w_k) - f(x_{\star})) 
\end{align}
$$


同样需要采用类似的技术先控制梯度的范数，


$$
\begin{align}
\text{Let } g_i(x) &= f_i(x)- f_i(x_{k}) -\nabla f_i(x_{k})^T(x- x_{k}) \\
0 = g_i(x_{k}) &\le \min_{\eta} [g_i(x- \eta \nabla g_i(x))] \\
&\le \min_{\eta} [g_i(x) - \eta \Vert g_i(x) \Vert_2^2 + \frac{1}{2} L \eta^2 \Vert \nabla g_i(x) \Vert_2^2] \\
&=g_i(x) - \frac{1}{2L} \Vert \nabla g_i(x) \Vert_2^2 \\
E [\Vert \nabla f_i(w_k) - \nabla f_i(x_{\star}) \Vert_2^2] &= E[\Vert \nabla g_i(w_k) \Vert_2^2] \\
&\le 2L E[g_i(w_k)- g_i(x_{k})] \\
&= 2LE[f_i(w_k) - f_i(x_{k} ) - \nabla f_i(x_{k})^T(w_k - x_{k})] \\
&=2L [f(w_k) - f(x_{k}) - \nabla f(x_k)^T(w_k - x_k)]
\end{align}
$$


在最终的证明之前需要一些技巧性较强的引理，首先控制 $g_k$ 的方差， 


$$
\begin{align}
E[\Vert g_k - \nabla f(x_k) \Vert_2^2] &= Var[ g_k] \\
&= Var[\nabla f_i(x_k) - \nabla f_i (w_k)] \\
&\le E[\Vert \nabla f_i(x_k) - \nabla f_i(w_k) \Vert_2^2] \\
&\le 2L(f(w_k) - f(x_{k}) - \nabla f(x_k)^T(w_k - x_k))
\end{align}
$$


对于 $z_k,z_{k+1}$ 之间的联系，与Lyapunov函数中对应的项联系起来


$$
\begin{align}
z_{k+1} &=\frac{1}{1+ \eta \sigma}(\eta \sigma x_k + z_k - \frac{\eta}{L} g_k) \\
g_k &=  \frac{L}{\eta} ( \eta \sigma x_k + z_k - (1+ \eta \sigma) z_{k+1}  ) \\
&= \mu x_k + \frac{L}{\eta} z_k  - (\frac{L}{\eta} + \mu) z_{k+1} \\
&=\frac{L}{\eta}(z_{k} -z_{k+1}) + \mu (x_k - z_{k+1}) \\
g_k^T(z_{k+1} - x_{\star}) &= (\frac{L}{\eta}(z_{k} -z_{k+1}) + \mu (x_k - z_{k+1}))^T(z_{k+1}-x_{\star}) \\
&= \frac{L}{\eta}(z_k - z_{k+1} )^T(z_{k+1} - x_{\star}) + \mu (x_k - z_{k+1})^T(z_{k+1} - x_{\star}) \\
&= \frac{L}{2\eta}( \Vert z_{k} - x_{\star} \Vert_2^2 - \Vert z_k- z_{k+1}  \Vert_2^2 - \Vert z_{k+1} - x_{\star} \Vert_2^2)  + \frac{\mu}{2} (\Vert x_k - x_{\star} \Vert_2^2- \Vert x_k - z_{k+1} \Vert_2^2 - \Vert z_{k+1} - x_\star \Vert_2^2)\\ 
&\le \frac{L}{2\eta}( \Vert z_{k} - x_{\star} \Vert_2^2 - \Vert z_k- z_{k+1}  \Vert_2^2 - \Vert z_{k+1} - x_{\star} \Vert_2^2)  + \frac{\mu}{2} (\Vert x_k - x_{\star} \Vert_2^2 - \Vert z_{k+1} - x_\star \Vert_2^2)\\ 
&= \frac{L}{2 \eta} \Vert z_k - x_{\star} \Vert_2^2 - (\frac{L}{2\eta} + \frac{\mu}{2}) \Vert z_{k+1} - x_{\star} \Vert_2^2 - \frac{L}{2 \eta} \Vert z_k - z_{k+1} \Vert_2^2 + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 \\
&=\frac{L}{2 \eta} \Vert z_k - x_{\star} \Vert_2^2 - \frac{L(1+ \eta \sigma)}{2\eta} \Vert z_{k+1} - x_{\star} \Vert_2^2 - \frac{L}{2 \eta} \Vert z_k - z_{k+1} \Vert_2^2 + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 \\
&=\frac{1}{1+\eta \sigma} \mathcal{Z_k} - \mathcal{Z_{k+1}}- \frac{L}{2 \eta} \Vert  z_{k+1} - z_k \Vert_2^2 + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 \\
\end{align}
$$



利用更新公式可以得到 $z_k,y_k$ 之间的关系，

$$
\begin{align}
y_{k+1} - x_k  &= \theta_1 (z_{k+1} - z_k) \\
\frac{L}{2\eta} \Vert z_{k+1} - z_k \Vert_2^2 + g_k^T(z_{k+1} - z_k) &= \frac{L}{2 \eta \theta_1^2} \Vert y_{k+1} - x_k \Vert_2^2 + \frac{1}{\theta_1} g_k^T(y_{k+1}-x_k) \\ 
&=\frac{L}{2 \eta \theta_1^2} \Vert y_{k+1} - x_k \Vert_2^2 + \frac{1}{\theta_1} \nabla f(x_k)^T(y_{k+1}-x_k) + \frac{1}{\theta_1}  (g_k - \nabla  f(x_k))^T(y_{k+1}-x_k) \\
&\ge \frac{L}{2 \eta \theta_1^2} \Vert y_{k+1} - x_k \Vert_2^2 + \frac{1}{\theta_1} (f(y_{k+1})- f(x_k) -\frac{L}{2 } \Vert y_{k+1} - x_k \Vert_2^2) + \frac{1}{\theta_1}  (g_k - \nabla f(x_k))^T(y_{k+1}-x_k) \\
&= \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k)) + (\frac{L}{2 \eta \theta_1^2} - \frac{L}{2\theta_1}) \Vert y_{k+1} - x_k \Vert_2^2 +\frac{1}{\theta_1} (g_k - \nabla f(x_k))^T(y_{k+1}-x_k) \\ 
&= \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k) +  \frac{L(1-\eta \theta_1)}{2\eta \theta_1} \Vert y_{k+1} - x_k \Vert_2^2 + (g_k - \nabla f(x_k))^T(y_{k+1}-x_k))  \\ 										
&\ge \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k) - \frac{\eta \theta_1}{2 L(1-\eta \theta_1)} \Vert g_k - \nabla f(x_k) \Vert_2^2 ) \\																								
\end{align}
$$



由于最终收敛性和更新公式密切相关，回顾更新公式，
$$
\begin{align}
x_k &= \theta_1 z_k + \theta_2 w_k + (1-\theta_1 - \theta_2) y_k \\
g_k &= \nabla f_i(x_k) - \nabla f_i (w_k) + \nabla f(w_k), i \sim \mathcal{U}(1..n) \\
z_{k+1} &=\frac{1}{1+ \eta \sigma}(\eta \sigma x_k + z_k - \frac{\eta}{L} g_k), \text{With } \sigma = \frac{1}{\kappa} \\ 
y_{k+1} &=x_k + \theta_1 (z_{k+1} - z_k) \\
w_{k+1} &= y_k \text{ With Prob. p}  \\
&=w_k \text{With Prob. } 1-p
\end{align}
$$


并且整理我们得到的不等式，


$$
\begin{align}
g_k^T(x_{\star} - z_{k+1}) +\frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 &\ge \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} +  \frac{L}{2 \eta} \Vert  z_{k+1} - z_k \Vert_2^2 \\
 \frac{L}{2 \eta} \Vert  z_{k+1} - z_k \Vert_2^2 + \nabla g_k^T(z_{k+1} - z_k)) &\ge \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k) - \frac{\eta \theta_1}{2 L(1-\eta \theta_1)} \Vert g_k - \nabla f(x_k) \Vert_2^2 ) \\
-E[\Vert g_k - \nabla f(x_k) \Vert_2^2] &\ge 2L(f(x_k) - f(w_{k}) - \nabla f(x_k)^T(x_k - w_k))
\end{align}
$$


在上述所有的准备之后就可以推出最终的收敛性结果，


$$
\begin{align}
x_k &= \theta_1 z_k + \theta_2 w_k + (1-\theta_1 - \theta_2) y_k \\ 
z_k &= \frac{1}{\theta_1} x_k  - \frac{\theta_2}{\theta_1} w_k - \frac{1- \theta_1 - \theta_2}{\theta_1} y_k\\ 
z_k - x_k &= \frac{1-\theta_1}{\theta_1} x_k  - \frac{\theta_2}{\theta_1} w_k - \frac{1- \theta_1 - \theta_2}{\theta_1} y_k\\  
&=  \frac{1-\theta_1}{\theta_1} x_k  - \frac{\theta_2}{\theta_1} w_k - \frac{1- \theta_1 - \theta_2}{\theta_1} y_k\\
&= \frac{\theta_2}{\theta_1}(x_k - w_k) + \frac{1- \theta_1 - \theta_2}{\theta_1} (x_k  - y_k)\\
f(x_{\star}) &\ge f(x_k) + \nabla f(x_k)^T(x_{\star}- x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 \\
&= f(x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +\nabla f(x_k)^T(x_{\star}- x_k) \\
&= f(x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +\nabla f(x_k)^T(x_{\star}- z_k) + \nabla f(x_k)^T (z_k - x_k) \\
&= f(x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +\nabla f(x_k)^T(x_{\star}- z_k) + \frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) + \frac{1-\theta_1 - \theta_2}{\theta_1} \nabla f(x_k)^T(x_k - y_k) \\
&\ge f(x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +\nabla f(x_k)^T(x_{\star}- z_k) + \frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) + \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) + \frac{\mu}{2} \Vert x_k - y_k \Vert_2^2)\\
&\ge f(x_k) + \frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +\nabla f(x_k)^T(x_{\star}- z_k) + \frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) + \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) )\\
&=f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) )+\frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) +\frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +E[\nabla g_k^T(x_{\star}- z_k)] \\
&=f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) )+\frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) +\frac{\mu}{2} \Vert x_k - x_{\star} \Vert_2^2 +E[\nabla g_k^T(x_{\star}- z_{k+1}) + \nabla g_k^T(z_{k+1} - z_k))] \\
&\ge f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1}(f(x_k) - f(y_k) )+\frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) +E[  \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} + \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k) - \frac{\eta \theta_1}{2 L(1-\eta \theta_1)} \Vert g_k - \nabla f(x_k) \Vert_2^2 ) ] \\
&\ge f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) )+\frac{\theta_2}{\theta_1}\nabla f(x_k)^T (x_k - w_k) +E[  \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} + \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k)) + \frac{\eta}{1-\eta \theta_1}(f(x_k) - f(w_{k}) - \nabla f(x_k)^T(x_k - w_k))   ] \\
&= f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) ) +E[  \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} + \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k)) + \frac{\eta}{1-\eta \theta_1}(f(x_k) - f(w_{k}))   ] ,\text{Set } \eta = \frac{\theta_2}{\theta_1(\theta_2+ 1)}\\
\end{align}
$$


希望得到Lyapunov函数的形式，


$$
\begin{align}
f(x_{\star}) &\ge f(x_k) +  \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_k) - f(y_k) ) +E[  \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} + \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_k)) + \frac{\theta_2}{\theta_1}(f(x_k) - f(w_{k}))   ] \\
&= f(x_k)  +(\frac{1-\theta_1- \theta_2}{\theta_1} +\frac{\theta_2}{\theta_1} + \frac{1}{\theta_1}) (f(x_k) - f(x_{\star})) + \frac{1-\theta_1 - \theta_2}{\theta_1} (f(x_{\star}) - f(y_k) ) +E[  \mathcal{Z_{k+1}}- \frac{1}{1+\eta \sigma} \mathcal{Z_k} + \frac{1}{\theta_1} (f(y_{k+1} ) - f(x_{\star})) + \frac{\theta_2}{ \theta_1}(f(x_{\star}) - f(w_{k}))   ]  \\ 
&= f(x_\star) - (1-\theta_1 - \theta_2) \mathcal{Y_k} +E[ \mathcal{Z_{k+1}} -\frac{1}{1+\eta \sigma} \mathcal{Z_k} +\mathcal{Y_{k+1}} - \frac{p}{1+\theta_1}\mathcal{W_k} ] \\
\end{align}
$$


整理后得到下式，


$$
E[\mathcal{Z_{k+1}} + \mathcal{Y_{k+1}}] \le \frac{1}{1+ \eta \sigma} \mathcal{Z_k} +(1-\theta_1 - \theta_2 ) \mathcal{Y_k} + \frac{p}{1+\theta_1} \mathcal{W_k}
$$


最后根据概率更新公式可以得到 $\mathcal{W_{k+1}}, \mathcal{W_{k}}$ 的递推关系，


$$
\begin{align}
E[\Phi_{k+1}] &= E[\mathcal{Z_{k+1}} + \mathcal{Y_{k+1}} + \mathcal{W}_{k+1}] \\
&= E[\mathcal{Z_{k+1}} + \mathcal{Y_{k+1}} + (1-p )\mathcal{W_k} + (1+\theta_1) \theta_2 \mathcal{Y_k}] \\
&\le  \frac{1}{1+ \eta \sigma} \mathcal{Z_k} +(1-\theta_1 (1- \theta_2 )) \mathcal{Y_k} + (1-\frac{p \theta_1}{1+\theta_1}) \mathcal{W_k} \\
&\le \max(\frac{1}{1+ \eta \sigma}, 1-\theta_1(1-\theta_2) , 1- \frac{p \theta_1}{1+ \theta_1}) \Phi_k \\
&= \max(1- \frac{1}{1 + 3\kappa \theta_1}, 1 - \frac{p \theta_1}{1+ \theta_1} ), \Phi_k, \text{Set } \theta_2 = \frac{1}{2} \\
&= \max(1- \frac{1}{1 + 3\kappa \theta_1}, 1 - \frac{\theta_1}{n(1+ \theta_1)} ), \Phi_k, \text{Set } p = \frac{1}{n} \\
\end{align}
$$


为了达到 $\epsilon$-最优解所需要的迭代次数 $k$ 和期望下梯度计算次数 $k'$ 满足，


$$
\begin{align}
k &\ge \max(1+ 3 \kappa \theta_1, n (1+ \frac{1}{\theta_1})) \log \frac{1}{\epsilon} \\
k' &= (1+ pn) k \\
&\ge 2 (1+ n+3 \kappa \theta_1 + \frac{n}{\theta_1}) \\
&= 2(1+n + 2\sqrt{3 \kappa n}) , \text{Let } \theta_1 = \sqrt{\frac{n}{3 \kappa}} \\
\end{align}
$$


也即需要的梯度计算次数为 $O((n + \sqrt {\kappa n}) \log \frac{1}{\epsilon} ) $, 

对于L-SVRG的结果，可以发现L-Katyusha的结果优于L-SVRG的结果，尤其针对于严重病态也即条件数 $\kappa$ 很大的问题。
