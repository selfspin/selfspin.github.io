---
title: 'Decentralized AGD'
toc: true
excerpt_separator: <!--more-->
tags: 
  - 优化
---



论文阅读笔记：[Decentralized Accelerated Proximal Gradient Descent](https://proceedings.neurips.cc/paper/2020/hash/d4b5b5c16df28e61124e13181db7774c-Abstract.html)



<!--more-->



文章关注于去中心化的带正则项的经验风险最小化问题，但此时每一个损失函数存储在不同的去中心化的分布式机器上，也即，

$$
\begin{align}
\min h(x) &= f(x) + r(x) =\frac{1}{m} \sum_{i=1}^m f_i(x) + r(x)
\end{align}
$$

其中 $f_i(x)$ 为每台机器上的损失函数，$r(x)$ 为共享的正则项，文章将 [Nesterov加速方法](https://truenobility303.github.io/Nesterov-Acceleration/) 用于该去中心化的分布式优化上，

假设每一个函数 $f_i(x)$ 满足 $L$- 光滑 和 $\mu$-强凸性质，而其方法首先基于FastMix算子，其效果是利用分布式机器之间的通信矩阵 $W$ 进行通信，使得每台机器中的向量与均值接近，且算法的收敛速度为线性收敛性，也即使用FastMix算子迭代 $k$ 次之后，得到的所有机器上的向量 $\mathbf{x}$ 满足，


$$
\Vert \mathbf{x} - \bar{\mathbf{x}} \Vert \le (1- \sqrt{1- \lambda_2(W)})^k \Vert \mathbf{x_0} - \bar{\mathbf{x}} \Vert
$$

取Nesterov加速方法中的一个特例，令 $\theta_0 = \sqrt{\frac{\mu}{L} }$, 则有 $\gamma_k = \mu,$ 更新公式变为，



$$
\begin{align}
y_k &= x_k + \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa}+1 } (x_k - x_{k-1}) ,\text{With } \kappa = \frac{L}{\mu} \\
x_{k+1} &= \textbf{prox}_{\eta,r }(y_k - \eta \nabla f(x)) ,\eta  = \frac{1}{L}\\
\end{align}
$$



将其更新公式用于去中心化的场景，此时一个关键的问题是每一台机器并不能获取全局函数 $f(x)$ 的梯度 $\nabla f(x)$ ,因此每一台机器必须记录自己的梯度并且使用FastMix算子使得每台机器的梯度记录值接近于均值，则该记录值近似于全局梯度，该技术称为 Gradient Tracking技术，因此算法如下，

$$
\begin{align}
\mathbf{x_{k+1}} &= \text{FastMix}(\textbf{prox}_{\eta,r}(\mathbf{y_k} - \eta \mathbf{s_k})) \\
\mathbf{y_{k+1}} &= \text{FastMix}(\mathbf{x_{k+1}} + \frac{\sqrt\kappa - 1}{\sqrt \kappa +1}(\mathbf{x_{k+1}} - \mathbf{x_k}) ) \\
\mathbf{s_{k+1}} &= \text{FastMix}(\mathbf{s_k} + \nabla f(\mathbf{y_{k+1}})- \nabla f(\mathbf{y_k}) )
\end{align}
$$



类似Nesterov加速方法，收敛性的证明依赖于如下定义的Lyapunov函数，


$$
\begin{align}
\mathcal{V_k} &= h(\bar x_k) - h(x_{\ast}) + \frac{\mu}{2} \Vert \bar v_k - x_{\ast} \Vert_2^2 \\
\text{With } \bar v_k &=\bar x_{k-1} + \sqrt{\kappa}(\bar x_{k}- \bar x_{k-1})
\end{align}
$$


算法的收敛性证明可以转化为上述Lyapunov函数的收敛。



本文计算的常数项可以与原论文不一致，但证明思路与收敛率是相同的。

## Lemmas

在正式的证明之前，需要如下一些引理的准备，用来控制与梯度及其中心相关的量，并且写成向量形式，



类似于近端梯度下降算法中的证明，需要定义广义梯度，


$$
\begin{align}
y_{k+1} &= \textbf{prox}(y_k - \eta \nabla f(y_k)) \\
&=y_k - \eta \tilde \nabla h(y_k)  \\
\text{Define } \tilde \nabla h(y_k) &= \frac{y_{k} - \textbf{prox}(y_k - \eta \nabla f(y_k))}{\eta} \\
&= L(x_k - \textbf{prox}(y_k - \frac{1}{L} \nabla f(y_k))) ,\text{Let } \eta = \frac{1}{L}
\end{align}
$$


算法需要对梯度进行控制，将对于标量形式的结论推广到向量形式，


$$
\begin{align}
\Vert \nabla f(\mathbf{y}) - \nabla f(\mathbf{x}) \Vert &\le L \Vert y -x \Vert ,\text{By } \Vert \nabla f(y) - \nabla f(x) \vert \le \Vert y - x \Vert \\
\end{align}
$$


对于梯度的平均值也有类似的结论，


$$
\begin{align}
\Vert \mathbf{1} \cdot\bar g  -  \mathbf{1} \cdot \nabla f(\bar y)\Vert &= \Vert \sum_{i=1}^m \frac{  \nabla f(y^{(i)}) -\nabla f(\bar y) }{\sqrt m} \Vert, \text{Let } \bar g = \frac{1}{m} \sum_{i=1}^m \nabla f(y^{(i)})\\
&\le \Vert  \nabla f(\mathbf{y}) - \mathbf{1} \cdot \nabla f(\bar y)\Vert, \text{By } \Vert \sum_{i=1}^m \mathbf{a}^{(i)} \Vert \le \sqrt{m} \Vert \mathbf{a} \Vert \\
&\le L \Vert \mathbf{y} - \mathbf{1} \cdot \bar y \Vert
\end{align}
$$


对于广义梯度，用到了 $\textbf{prox}$ 算子的非拓展性，也即,


$$
\begin{align}
\text{By } \Vert \textbf{prox}(\mathbf{x}) - \textbf{prox}(\mathbf{y}) \Vert &\le \Vert \mathbf{x} - \mathbf{y} \Vert \\
\Vert \tilde \nabla h(\mathbf{x}) - \tilde \nabla h(\mathbf{y}) \Vert &=  L \Vert (\mathbf{x} - \mathbf{y}) -(\textbf{prox}(\mathbf{x} - \eta \nabla f(\mathbf{x}) )) - \textbf{prox}(\mathbf{y} - \eta \nabla f(\mathbf{y})) \Vert \\
&\le L \Vert \mathbf{x} - \mathbf{y} \Vert + L \Vert \textbf{prox}(\mathbf{x} - \eta \nabla f(\mathbf{x}) )) - \textbf{prox}(\mathbf{y} - \eta \nabla f(\mathbf{y})) \Vert  \\
&\le L \Vert \mathbf{x} - \mathbf{y} \Vert + L \Vert (\mathbf{x} - \mathbf{y}) - \eta(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})) \Vert \\
&\le 2 L \Vert \mathbf{x} - \mathbf{y} \Vert + \Vert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \Vert \\
&\le 3L \Vert \mathbf{x} - \mathbf{y} \Vert
\end{align}
$$


类似地对于与 $\textbf{prox}$ 算子相关的平均值，


$$
\begin{align}
\Vert \mathbf{1} \cdot \bar p -\mathbf{1} \cdot \textbf{prox}(\bar x)  \Vert &= \Vert \sum_{i=1}^m \frac{ \textbf{prox} (x^{(i)})-\textbf{prox}(\bar x) }{\sqrt m} \Vert, \text{Let } \bar p = \frac{1}{m} \sum_{i=1}^m \textbf{prox}(x^{(i)})\\
&\le \Vert  \textbf{prox} (\mathbf{x}) - \mathbf{1} \cdot \textbf{prox} (\bar x)\Vert, \text{By } \Vert \sum_{i=1}^m \mathbf{a}^{(i)} \Vert \le \sqrt{m} \Vert \mathbf{a} \Vert \\
&\le \Vert \mathbf{x} - \mathbf{1} \cdot \bar x \Vert
\end{align}
$$


由于算法中采用了 Gradient Tracking 的技术，采用 $s_k$ 相关的量估计真正的梯度，需要估计其和真实的广义梯度之间的差距，

算法用巧妙之处在于使用的FastMix算子并不会改变平均值，因此根据递推公式我们可以知道梯度的平均值保持一定， $\bar s_k = \bar g_k$ 


$$
\begin{align}
\mathbf{s_{k+1}} &= \text{FastMix}(\mathbf{s_k} + \nabla f(\mathbf{y_{k+1}})- \nabla f(\mathbf{y_k}) ) \\
\bar s_{k+1} &= \bar s_k + \bar g_{k+1} - \bar g_k ,\text{Let } \bar g_k = \frac{1}{m } \sum_{i=1}^m \nabla f(y_k) \\
\text{We know } \bar s_0 &= \bar g_0, \text{Then by induction } \bar s_k = \bar g_k
\end{align}
$$


利用该重要的性质可以得到广义梯度的估计量和真实广义梯度的估计量之间的差距，证明的时候先利用和均值的差距，


$$
\begin{align}
\text{Define } \textbf{G}_k &= \frac{\textbf{y}_k - \textbf{prox}(\textbf{y}_k - \eta \textbf{s}_k)}{\eta} \\
\Vert \eta \mathbf{1} \cdot \bar G_k - \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert &=  \Vert \sum_{i=1}^m  \frac{\eta G_k^{(i)} - \eta \tilde \nabla h(\bar y_k)}{\sqrt m} \Vert \\
&\le  \Vert  \eta \mathbf{G_k} -  \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert \\
&=  \Vert (\mathbf{y}_k - \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k})) - \mathbf{1} \cdot (\bar y_k - \textbf{prox}(\bar y_k - \eta \nabla f(\bar y_k)) \Vert \\
&= \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + \Vert \textbf{prox}(\mathbf{y_k} -\eta \mathbf{s_k}) - \textbf{prox}(\mathbf{1} \cdot \bar y_k - \eta  \mathbf{1} \cdot \nabla f(\bar y_k)) \Vert \\
&\le \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + \Vert (\mathbf{y_k} -  \mathbf{1} \cdot \bar y_k) - \eta( \mathbf{s_k} - \mathbf{1} \cdot \nabla f(\bar y_k)) \Vert\\
&\le 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert  \mathbf{s_k}- \mathbf{1} \cdot \nabla f(\bar y_k) \Vert \\
&=2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert  (\mathbf{s_k}- \mathbf{1} \cdot \bar s_k) - (\mathbf{1} \cdot \nabla f(\bar y_k) -\mathbf{1} \cdot \bar s_k) \Vert \\
&\le 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \eta\Vert \mathbf{1} \cdot \bar s_k - \mathbf{1} \cdot \nabla f(\bar y_k) \Vert \\
&= 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \eta \Vert \mathbf{1} \cdot \bar g_k - \mathbf{1} \cdot \nabla f(\bar y_k) \Vert ,\text{By }\bar g_k = \bar s_k \\
&\le 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \eta\Vert  \nabla f(\mathbf{y_k})  - \mathbf{1} \cdot \nabla f(\bar y_k) \Vert \\
&\le 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \eta\Vert  \nabla f(\mathbf{y_k})  - \mathbf{1} \cdot \nabla f(\bar y_k) \Vert \\
&\le 2 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \Vert  \mathbf{y_k}  - \mathbf{1}  \cdot \bar y_k \Vert ,\text{By } \eta = \frac{1}{L}\\
&=3 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$



在均值的差距可以被控制的基础上，整个广义梯度的估计量与真实广义梯度的差距也理应被控制，



首先有，
$$
\begin{align}
 \Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \bar G_k \Vert 
&\le \Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \bar G_k \Vert  \\
&= \Vert (\mathbf{y_k} - \mathbf{1} \cdot \bar y_k) - (\textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k}) - \mathbf{1} \cdot \bar p_k) \Vert \\
&\le \Vert \mathbf{y_k}  - \mathbf{1} \cdot \bar y_k \Vert+ \Vert \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k}) -  \mathbf{1} \cdot \bar p_k \Vert \\
&=\Vert \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k}) - \mathbf{1} \cdot \bar p_k \Vert +  \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert  ,\text{Define } \bar p_k = \frac{1}{m} \sum_{i=1}^m \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k})\\  
&\le \Vert \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k}) -\mathbf{1} \cdot \textbf{prox}(\bar y_k - \eta \bar s_k) \Vert + \Vert \mathbf{1} \cdot \textbf{prox}(\bar y_k - \eta \bar s_k) - \mathbf{1} \cdot \bar p_k \Vert + \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert  \\
&\le 2 \Vert \textbf{prox}(\mathbf{y_k} - \eta \mathbf{s_k}) -\mathbf{1} \cdot \textbf{prox}(\bar y_k - \eta \bar s_k) \Vert  +\Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert \\
&\le 2 \Vert (\mathbf{y_k}- \mathbf{1} \cdot \bar y_k) - (\eta \mathbf{s_k} - \eta \bar s_k) \Vert +  \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert \\
&\le 3 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$


因此，


$$
\begin{align}
\Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert &= \Vert (\eta \mathbf{G_k} - \eta \mathbf{1} \cdot\bar G_k) - (\eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) - \eta \mathbf{1} \cdot\bar G_k) \Vert \\
&\le \Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \bar G_k \Vert + \Vert \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) - \eta \mathbf{1} \cdot \bar G_k \Vert \\
&\le 6 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 3 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$



对于梯度 $\nabla f(x)$, Gradient Tracking得到的估计量也满足差距很小的性质，



$$
\begin{align}
\Vert \mathbf{s_k} - \nabla f(\mathbf{x_k}) \Vert &\le \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \Vert \nabla f(\mathbf{x_k}) - \mathbf{1} \cdot \bar s_k \Vert \\
&\le \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert  + \Vert \nabla f(\mathbf{x_k}) - \mathbf{1}\cdot \nabla f(\bar x_k) \Vert + \Vert \mathbf{1} \cdot \bar s_k - \mathbf{1} \cdot \nabla f(\bar x_k) \Vert \\
&=  \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \Vert \nabla f(\mathbf{x_k}) - \mathbf{1}\cdot \nabla f(\bar x_k) \Vert + \Vert \mathbf{1} \cdot \bar g_k - \mathbf{1} \cdot \nabla f(\bar x_k) \Vert ,\text{By } \bar g_k = \bar s_k\\
&\le \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + 2\Vert \nabla f(\mathbf{x_k}) - \mathbf{1}\cdot \nabla f(\bar x_k) \Vert \\
&\le  \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert  + 2L \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert \\
\end{align}
$$



## Bound for Consensus Error



基于上述的引理，本节证明使用FastMix操作后的向量距离均值的范数可以被控制住，

基于FastMix可以达到任意精度 $\rho$ 使得向量的每个元素都收敛于均值，也即


$$
\begin{align}
\Vert \text{FastMix}(\mathbf{x}) - \mathbf{1} \cdot \bar x \Vert \le \rho \Vert \mathbf{x} - \mathbf{1} \cdot \bar x \Vert 
\end{align}
$$

我们希望递推地控制住 $\mathbf{y_k}, \mathbf{s_k}, \mathbf{x_k}$ 离中心或者均值的距离，对于 $\mathbf{x_{k}}$ ,



$$
\begin{align}
\mathbf{x_{k+1}} &= \text{FastMix}(\textbf{prox}_{\eta,r}(\mathbf{y_k} - \eta \mathbf{s_k})) \\
\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert &\le \rho \Vert \textbf{prox}_{\eta,r}(\mathbf{y_k} - \eta \mathbf{s_k}) - \bar p_k \Vert ,\text{With } \bar p_k = \frac{1}{m} \sum_{i=1}^m  \textbf{prox}_{\eta,r}(\mathbf{y_k}^{(i)} - \eta \mathbf{s_k}^{(i)}) \\
&\le \rho \Vert \textbf{prox}_{\eta,r}(\mathbf{y_k} - \eta \mathbf{s_k}) -\textbf{prox}_{\eta,r}(\mathbf{1} \cdot \bar y_k - \eta \mathbf{1} \cdot \bar s_k) \Vert + \rho \Vert \textbf{prox}_{\eta,r}(\mathbf{1} \cdot \bar y_k - \eta \mathbf{1} \cdot \bar s_k) -\bar p_k \Vert \\
& \le 2 \rho \Vert \textbf{prox}_{\eta,r}(\mathbf{y_k} - \eta \mathbf{s_k}) -\textbf{prox}_{\eta,r}(\mathbf{1} \cdot \bar y_k - \eta \mathbf{1} \cdot \bar s_k) \Vert \\
& \le 2 \rho \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + 2 \rho \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert 
\end{align}
$$



对于 $\mathbf{y_k}$, 同样利用更新公式以及FastMix算子的性质，



$$
\begin{align}
\mathbf{y_{k+1}} &= \text{FastMix}(\mathbf{x_{k+1}} + \frac{\sqrt\kappa - 1}{\sqrt \kappa +1} (\mathbf{x_{k+1}} - \mathbf{x_k}) )\\
\Vert \mathbf{y_{k+1}} - \mathbf{1} \cdot \bar y_{k+1} \Vert  &\le \rho \Vert \mathbf{x_{k+1}} + \frac{\sqrt\kappa - 1}{\sqrt \kappa +1} (\mathbf{x_{k+1}} - \mathbf{x_k}) - \mathbf{1} \cdot(\bar x_{k+1} + \frac{\sqrt\kappa - 1}{\sqrt \kappa +1} (\bar x_{k+1} - \bar x_k)) \Vert\\
&\le \frac{2 \sqrt \kappa }{\sqrt \kappa +1} \rho\Vert \mathbf{x_{k+1}} - \mathbf{1} \cdot\bar x_{k+1} \Vert + \frac{\sqrt{\kappa} -1}{\sqrt{\kappa} +1} \rho \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert \\
&\le 2 \rho \Vert \mathbf{x_{k+1}} - \mathbf{1} \cdot\bar x_{k+1} \Vert+ \rho \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert \\ 
&\le 4 \rho^2 \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + 4 \rho^2 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \rho \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert 
\end{align}
$$



对于 $\mathbf{s_k}$, 同样利用更新公式以及FastMix算子的性质，



$$
\begin{align}
\mathbf{s_{k+1}} &= \text{FastMix}(\mathbf{s_k} + \nabla f(\mathbf{y_{k+1}})- \nabla f(\mathbf{y_k}) ) \\
\Vert \mathbf{s_{k+1}} - \mathbf{1} \cdot \bar s_{k+1} \Vert &\le \rho \Vert (\mathbf{s_k} + \nabla f(\mathbf{y_{k+1}})- \nabla f(\mathbf{y_k})) - (\mathbf{1} \cdot \bar s_k + \bar g_{k+1} - \bar g_k) \Vert \\
& \le \rho \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \rho \Vert (\nabla f(\mathbf{y_{k+1}}) - \nabla f(\mathbf{y_k})) - (\bar g_{k+1} - \bar g_k) \Vert \\
&\le \rho \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \rho \Vert \nabla f(\mathbf{y_{k+1}})- \nabla f(\mathbf{y_k}) \Vert \\
&\le \rho \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + L\rho\Vert \mathbf{y_{k+1}}- \mathbf{y_k} \Vert \\
&\le \rho \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + L\rho\Vert \mathbf{y_{k+1}}- \mathbf{1} \cdot \bar y_{k+1} \Vert  + L \rho \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L \rho \Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert\\
&\le \rho \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert  + L \rho \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L \rho \Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert + L\rho(4 \rho^2 \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + 4 \rho^2 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \rho \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert )\\
&= (\rho+4 L\rho^3 \eta) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + (L \rho + 4 L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L\rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert + L \rho \Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert \\
\end{align}
$$



仍然留下一项 $\Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert$ 需要化简，


$$
\begin{align}
\Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert &= \Vert \frac{2 \sqrt \kappa}{\sqrt \kappa +1} \mathbf{1} \cdot \bar x_{k+1} + \frac{\sqrt \kappa -1}{\sqrt \kappa + 1} \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot \bar y_k \Vert \\
&=\Vert \frac{2 \sqrt{\kappa}}{\sqrt \kappa +1} \mathbf{1} \cdot \bar p_k + \frac{\sqrt \kappa -1}{\sqrt \kappa + 1} \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot \bar y_k \Vert ,\text{With } \bar p_k = \frac{1}{m} \sum_{i=1}^m  \textbf{prox}_{\eta,r}(\mathbf{y_k}^{(i)} - \eta \mathbf{s_k}^{(i)}) \\
&= \Vert \frac{2 \sqrt{\kappa}}{\sqrt \kappa +1} \mathbf{1} \cdot (\bar p_k - \bar y_k) + \frac{\sqrt \kappa -1}{\sqrt \kappa + 1} \mathbf{1} \cdot (\bar x_k - \bar y_k) \Vert \\
&\le \Vert \frac{2 \sqrt{\kappa}}{\sqrt \kappa +1} \mathbf{1} \cdot \eta\bar G_k \Vert + \Vert \frac{\sqrt \kappa -1}{\sqrt \kappa + 1} \mathbf{1} \cdot (\bar x_k - \bar y_k) \Vert ,\text{With } \eta \bar G_k = \bar y_k - \bar p_k\\
&\le \frac{2 \sqrt{\kappa}}{\sqrt \kappa +1} \Vert \mathbf{1} \cdot \eta \bar G_k -  \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert + \frac{2 \sqrt{\kappa}}{\sqrt \kappa +1} \Vert \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert+ \frac{\sqrt \kappa -1}{\sqrt \kappa +1}\Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert + \frac{\sqrt \kappa -1}{\sqrt \kappa +1} \Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert \\
&\le  2\Vert \mathbf{1} \cdot \eta \bar G_k -  \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert +  2\Vert \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert+ \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert +  \Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert \\
&= 2\Vert \mathbf{1} \cdot \eta \bar G_k -  \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) \Vert +  2\Vert \mathbf{1} \cdot \eta \tilde \nabla h(\bar y_k) - \mathbf{1} \cdot \eta \tilde \nabla h(x_{\ast}) \Vert+ \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert +  \Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert \\
&\le 6 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + 2 \eta L\Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert + \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert +  \Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert \\
&= 6 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert +  3\Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert \\
\end{align}
$$


根据更新公式和Lyapunov函数的定义可以得到如下的关系式，



$$
\begin{align}
\mathcal{V_k} &= h(\bar x_k) - h(x_{\ast}) + \frac{\mu}{2} \Vert \bar v_k - x_{\ast} \Vert_2^2 \\
\text{With } \bar v_k &=\bar x_{k-1} + \sqrt{\kappa}(\bar x_{k}- \bar x_{k-1}) \\
(\sqrt \kappa+1) \bar y_k  &= 2 \sqrt \kappa \bar x_k - (\sqrt \kappa -1 )\bar x_{k-1} \\
&=\sqrt \kappa \bar x_k + \bar x_{k-1} + \sqrt \kappa (\bar x_k - \bar x_{k-1} ) \\
&= \sqrt \kappa \bar x_k + \bar v_k \\
\sqrt \kappa (\bar y_k - \bar x_k) &= \bar v_k - \bar y_k
\end{align}
$$



因此可以利用函数的 $\mu$-强凸性质和最优点的性质，


$$
\begin{align}
\frac{\mu}{2}\Vert \bar y_k - x_{\ast} \Vert^2 &= \Vert \frac{\sqrt \kappa }{\sqrt \kappa +1}(\bar x_k - x_{\ast}) + \frac{1}{\sqrt \kappa +1} (\bar v_k - x_{\ast})\Vert^2 \\
&\le  \frac{\mu}{2}[\frac{2\sqrt \kappa }{\sqrt \kappa +1} \Vert \bar x_k - x_{\ast} \Vert^2 +  \frac{2}{\sqrt \kappa +1}  \Vert \bar v_k - x_{\ast}\Vert^2] \\
&\le 2(\frac{\mu}{2} \Vert \bar x_k - x_{\ast} \Vert^2 + \frac{\mu}{2} \Vert \bar v_k - x_{\ast} \Vert^2) \\
&\le 2 (h(\bar x_k) - h(x_{\ast}) + \frac{\mu}{2} \Vert \bar v_k - x_{\ast } \Vert^2) \\
&=2 \mathcal{V_k} 
\end{align}
$$


化简后就可以得到，


$$
\begin{align}
\Vert \bar y_k - x_{\ast} \Vert &\le  2 \sqrt{\frac{\mathcal{V_k}}{\mu}} \\
\Vert \bar x_k - x_{\ast} \Vert &\le  \sqrt{\frac{2\mathcal{V_k}}{\mu}} \le 2 \sqrt{\frac{\mathcal{V_k}}{\mu}}\\
\Vert \bar v_k - x_{\ast} \Vert &\le  \sqrt{\frac{2\mathcal{V_k}}{\mu}} \le 2 \sqrt{\frac{\mathcal{V_k}}{\mu}}\\
\Vert \mathbf{1} \cdot\bar y_k - \mathbf{1} \cdot x_{\ast} \Vert &\le  2 \sqrt{\frac{m\mathcal{V_k}}{\mu}} \\
\Vert \mathbf{1} \cdot\bar x_k - \mathbf{1} \cdot x_{\ast} \Vert &\le  2 \sqrt{\frac{m\mathcal{V_k}}{\mu}} \\
\end{align}
$$


将上述式子代入之前的式子就可以得到，


$$
\begin{align}
\Vert \mathbf{s_{k+1}} - \mathbf{1} \cdot \bar s_{k+1} \Vert & \le (\rho+4 L\rho^3 \eta) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + (L \rho + 4 L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L\rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert + L \rho \Vert \mathbf{1} \cdot \bar y_{k+1} - \mathbf{1} \cdot \bar y_k \Vert \\
&\le (\rho+4 L\rho^3 \eta) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + (L \rho + 4 L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L\rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert + L \rho (6 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert +  3\Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot x_{\ast} \Vert) \\
&= (3\rho + 4 L\rho^3 \eta ) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert +(7 L\rho + 4L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L \rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert +  L \rho \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert + 3 L\rho \Vert \mathbf{1} \cdot \bar y_k - \mathbf{1} \cdot \bar y_k \Vert \\
&\le (3\rho + 4 L\rho^3 ) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert +(7 L\rho + 4L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L\rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert +   8L \rho \sqrt{\frac{m\mathcal{V_k}}{\mu}} \\
\end{align}
$$



回归我们得到的关于中心化的不等式，这些不等式告诉我们此时去中心化的算法的行为可以接近中心化的算法的行为，


$$
\begin{align}
\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert 
& \le 2 \rho \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + 2 \rho \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert \\
\Vert \mathbf{y_{k+1}} - \mathbf{1} \cdot \bar y_{k+1} \Vert  
&\le 4 \rho^2 \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + 4 \rho^2 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert + \rho \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert \\
\Vert \mathbf{s_{k+1}} - \mathbf{1} \cdot \bar s_{k+1} \Vert 
&\le (\rho + 4 L\rho^3 \eta + 2 \eta) \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert +(7 L\rho + 4L \rho^3) \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k \Vert + L \rho^2 \Vert \mathbf{x_k} - \bar x_k \Vert +   8L \rho \sqrt{\frac{m\mathcal{V_k}}{\mu}} \\ 
\end{align}
$$


最终定义向量 $\mathbf{z_k} $ 涵盖了我们所关心的距离中心的距离，可以得到


$$
\begin{align}
\begin{pmatrix}
\Vert \mathbf{x_{k+1}} - \mathbf{1} \cdot \bar x_{k+1} \Vert \\
\Vert \mathbf{y_{k+1}} - \mathbf{1} \cdot \bar y_{k+1} \Vert \\
\Vert \mathbf{s_{k+1}} - \mathbf{1} \cdot \bar s_{k+1} \Vert
\end{pmatrix}
\le
\begin{pmatrix}
0 & 2\rho  & 2 \rho \eta \\
\rho & 4 \rho^2 & 4\rho^2 \eta \\
L\rho^2 & 7 L\rho + 4L \rho^3 & 3 \rho + 4 \rho^3  
\end{pmatrix}
\begin{pmatrix}
\Vert \mathbf{x_{k}} - \mathbf{1} \cdot \bar x_{k} \Vert \\
\Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_{k} \Vert \\
\Vert \mathbf{s_{k}} - \mathbf{1} \cdot \bar s_{k} \Vert
\end{pmatrix}
+
8L \rho \sqrt m 
\begin{pmatrix}
0 \\
0 \\
\sqrt{\frac{\mathcal{V_k}}{\mu}}
\end{pmatrix}
\end{align}
$$


可以看到最终距离中心的距离都可以被 $\rho$ 所控制，定义 $A$ 为对应的矩阵，并且递推可以得到，


$$
\begin{align}
\mathbf{z_{k+1}} &\le A \mathbf{z_k} + 8L \rho \sqrt m  (0,0,\sqrt{\frac{\mathcal{V_k}}{\mu}})^T \\
\mathbf{z_{k+1}} &\le A^{k+1} \mathbf{z_0} + 8L \rho \sqrt m \sum_{i=0}^k A^{k-i} (0,0,\sqrt{\frac{\mathcal{V_i}}{\mu}})^T
\end{align}
$$

而由于矩阵 $A$ 为与 $\rho$ 相关的矩阵， 根据FastMix算子的性质，$\rho$ 的大小可以任意控制，后面将控制 $A$ 的谱半径小于 $\frac{1}{2}$ ，只需，


$$
\begin{align}
\text{Let } \rho &\le 7+ 6L + \frac{3}{L} \\
\text{Then }  \Vert A \Vert_2 &\le \Vert A \Vert_F \le \sum_{i,j} A_{ij}  \le \frac{1}{2}
\end{align}
$$


## Bound for Lyapunov Function



类似Nesterov加速方法的证明，同样必须证明Lyapunov函数的递推收敛，但在去中心化分布式优化的场景下，会出现和 $\mathbf{z_k}$ 有关的量，

对于每一台机器上的向量，运用完全类似的证明，但此时下降的值为Gradient Tracking产生的 $\mathbf{s_k}$, 因此误差项会多一些，根据次梯度


$$
\begin{align}
\mathbf{p_k} &=\mathbf{y_k} - \eta \mathbf{G_k} = \textbf{prox}_{\eta,h}(\mathbf{y_k} - \eta \mathbf{s_k})  \\
\mathbf{G_k} &\in \mathbf{s_k} + \partial h(\mathbf{x_k} - \eta \mathbf{G_k}) \\
\end{align}
$$


同样地进行一番奇妙的凸组合，一方面,



$$
\begin{align}
h(p_k^{(i)}) & = h(y_k^{(i) } - \eta G_k^{(i)}) \\
&= f(y_k^{(i) } - \eta G_k^{(i)}) + r(y_k^{(i) } - \eta G_k^{(i)}) \\
&\le f(y_k^{(i)}) - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 +  r(y_k^{(i) } - \eta G_k^{(i)}) \\
&\le f(y_k^{(i)}) - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 + r(y_k^{(i)}) - (G_k^{(i)} - s_k^{(i)})^T(\bar x_k - y_k^{(i)} + \eta G_k^{(i)}) \\
& \le f(\bar x_k) - \nabla f(y_k^{(i)})^T(\bar x_k - y_k^{(i)}) - \frac{\mu}{2} \Vert \bar x_k - y_k^{(i)} \Vert^2 - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 + r(y_k^{(i)}) - (G_k^{(i)} - s_k^{(i)})^T(\bar x_k - y_k^{(i)} + \eta G_k^{(i)}) \\
&=h(\bar x_k) - \frac{\mu}{2} \Vert \bar x_k - y_k^{(i)} \Vert^2 + (s_k^{(i)} - G_k^{(i)}- \nabla f(y_k^{(i)}))^T(\bar x_k - y_k^{(i)} + \eta G_k^{(i)}) - \frac{1}{2L} \Vert G_k^{(i)} \Vert^2
 \end{align}
$$



另一方面，


$$
\begin{align}
h(p_k^{(i)}) & = h(y_k^{(i) } - \eta G_k^{(i)}) \\
&= f(y_k^{(i) } - \eta G_k^{(i)}) + r(y_k^{(i) } - \eta G_k^{(i)}) \\
&\le f(y_k^{(i)}) - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 +  r(y_k^{(i) } - \eta G_k^{(i)}) \\
&\le f(y_k^{(i)}) - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 + r(y_k^{(i)}) - (G_k^{(i)} - s_k^{(i)})^T(x_{\ast} - y_k^{(i)} + \eta G_k^{(i)}) \\
& \le f(x_{\ast} - \nabla f(y_k^{(i)})^T(x_{\ast} - y_k^{(i)}) - \frac{\mu}{2} \Vert x_{\ast} - y_k^{(i)} \Vert^2 - \eta \nabla f(y_k^{(i)})^T G_k^{(i)} + \frac{L \eta^2}{2} \Vert G_k^{(i)} \Vert^2 + r(y_k^{(i)}) - (G_k^{(i)} - s_k^{(i)})^T(x_{\ast} - y_k^{(i)} + \eta G_k^{(i)}) \\
&=h(x_{\ast}) - \frac{\mu}{2} \Vert x_{\ast} - y_k^{(i)} \Vert^2 + (s_k^{(i)} - G_k^{(i)}- \nabla f(y_k^{(i)}))^T(x_{\ast} - y_k^{(i)}+ \eta G_k^{(i)}) -\frac{1}{2L} \Vert G_k^{(i)} \Vert^2 
\end{align}
$$


进行凸组合过后得到下面的式子，为了方便令 $\alpha = \frac{1}{\sqrt{\kappa}}$， 


$$
\begin{align}
h(p_k^{(i)}) \le \alpha h(x_{\ast})  +(1-\alpha) h(\bar x_k) - \frac{\mu}{2} \alpha \Vert x_{\ast} - y_k^{(i)} \Vert^2- \frac{\mu}{2} (1-\alpha) \Vert \bar x_k - y_k^{(i)} \Vert^2    + (s_k^{(i)} - G_k^{(i)}- \nabla f(y_k^{(i)}))^T(\alpha x_{\ast} + (1-\alpha) \bar x_{k} + \eta G_k^{(i)}- y_k^{(i)}) - \frac{L}{2} \Vert G_k^{(i)} \Vert^2 
\end{align}
$$



对于所有机器上的函数，利用目标函数 $h(x)$ 的凸性并且求平均值，


$$
\begin{align}
h(\bar x_{k+1}) - h(x_{\ast}) \le & \frac{1}{m} \sum_{i=1}^m h(p_k^{(i)}) - h(x_{\ast}) \\
\le &  (1-\alpha) (h(\bar x_k) - h(x_{\ast})) + \frac{1}{m}\sum_{i=1}^m [- \frac{\mu}{2} \alpha \Vert x_{\ast} - y_k^{(i)} \Vert^2- \frac{\mu}{2} (1-\alpha) \Vert \bar x_k - y_k^{(i)} \Vert^2    + (s_k^{(i)} - G_k^{(i)}- \nabla f(y_k^{(i)}))^T(\alpha x_{\ast} + (1-\alpha) \bar x_{k} + \eta G_k^{(i)}- y_k^{(i)}) -\frac{1}{2L} \Vert G_k^{(i)} \Vert^2] \\
\le &  (1-\alpha) (h(\bar x_k) - h(x_{\ast})) - \frac{\mu}{2} \alpha \Vert x_{\ast} - \bar y_k \Vert^2- \frac{\mu}{2} (1-\alpha) \Vert \bar x_k - \bar y_k \Vert^2  + \frac{1}{m} \sum_{i=1}^m [(s_k^{(i)} - G_k^{(i)}- \nabla f(y_k^{(i)}))^T(\alpha x_{\ast} + (1-\alpha) \bar x_{k} +\eta G_k^{(i)}- y_k^{(i)}) -\frac{1}{2L} \Vert G_k^{(i)} \Vert^2] \\
\end{align}
$$




下面推导 [Nesterov加速方法](https://truenobility303.github.io/Nesterov-Acceleration/) 的证明过程中的另一个核心的式子，由于FastMix算子不改变均值，因此对于均值与原方法有相同的结论

$$
\begin{align}
\bar v_{k+1} &= \frac{1}{\alpha} (\bar y_k - (1- \alpha) \bar x_k) - \frac{\alpha}{\mu} \bar G_k \\
\mu \bar v_{k+1} &= \mu \bar v_k + \mu \alpha (\bar y_k - \bar v_k) - \alpha \bar G_k \\
\frac{\mu}{2} \Vert \bar v_{k+1} - x_{\ast} \Vert^2 &= \frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k - x_{\ast} \Vert^2 - \frac{\alpha(1-\alpha)}{2} - \alpha \bar G_k^T(\bar v_k - x_{\ast} + \alpha(\bar y_k - \bar v_k)) \\
&\le \frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k - x_{\ast} \Vert^2 + \alpha \bar G_k^T(\bar v_k - x_{\ast} + \alpha(\bar y_k - \bar v_k)) \\
&= \frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k - x_{\ast} \Vert^2 +  \bar G_k^T(\alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k) \\
\end{align}
$$



由于此处的 $G_k$ 仅仅是广义梯度的估计量，需要利用两者差距可以被控制住的性质，



$$
\begin{align}
\frac{\mu}{2} \Vert \bar v_{k+1} - x_{\ast} \Vert^2 &\le
\frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k  -x_{\ast} \Vert^2 +  \bar G_k^T(\alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k) \\
&\le \frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2  + \frac{\mu \alpha}{2} \Vert \bar y_k  -x_{\ast} \Vert^2 +  \bar G_k^T(\alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k) \\
&\le \frac{\mu(1-\alpha)}{2} \Vert \bar v_k -x_{\ast} \Vert^2 + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k  -x_{\ast} \Vert^2 +  \tilde \nabla h(\bar y_k)^T(\alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k) + \Vert \bar G_k - \tilde \nabla h(\bar y_k) \Vert \Vert \alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k \Vert \\
\end{align}
$$



利用上述的关键式子，结合Cauthy不等式，以及Consensus Error相关的引理中告诉我们， 广义梯度的估计量 $G_k$ 距离真实的广义梯度 $\tilde \nabla h(y_k)$ 差距不大，而Gradient Tracking追踪的梯度 $\mathbf{s_k}$ 也和真实的梯度 $\nabla f(y_k)$ 差距不大,


$$
\begin{align}
\mathcal{V_{k+1}} =&  h(\bar x_{k+1}) - h(x_{\ast}) + \frac{\mu}{2} \Vert \bar v_{k+1} - x_{\ast} \Vert_2^2 \\
\le &(1- \alpha)(h(\bar x_k) - h(x_{\ast} ) + \frac{\mu}{2} \Vert \bar v_k - x_{\ast} \Vert^2) \\
\quad &     - \frac{1}{2Lm} \Vert \mathbf{G_k} \Vert^2 - \frac{\mu}{2} \alpha \Vert x_{\ast} - \bar y_k \Vert^2- \frac{\mu}{2} \alpha \Vert x_{\ast} - \bar y_k \Vert^2\\
\quad & +\frac{1}{m} \sum_{i=1}^m (s_k^{(i)} - \nabla f(y_k^{(i)})+  \tilde \nabla h(\bar y_k)-G_k^{(i)})^T(\alpha x_{\ast} + (1-\alpha) \bar x_{k} +\eta G_k^{(i)}- y_k^{(i)}) \\
\quad & + \frac{1}{m} \sum_{i=1}^m \tilde \nabla h(\bar y_k)^T( y_k^{(i)} - \bar y_k -\eta G_k^{(i)})  \\
\quad & + \frac{1}{2L} \Vert \bar G_k \Vert^2 + \frac{\mu \alpha}{2} \Vert \bar y_k  -x_{\ast} \Vert^2 + \Vert \bar G_k - \tilde \nabla h(\bar y_k) \Vert \Vert \alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k \Vert \\ 
\le & (1-\alpha)\mathcal{V_k} + \frac{1}{2Lm} \Vert \mathbf{1} \cdot \bar G_k \Vert^2- \frac{1}{2Lm} \Vert \mathbf{G_k} \Vert^2 \\
\quad & + \frac{1}{m}\Vert \mathbf{s_k} - \nabla f(\mathbf{y_k} ) + \mathbf{1} \cdot \tilde \nabla h(\bar y_k) - \mathbf{G_k} \Vert \Vert \alpha \mathbf{1} \cdot x_{\ast} + (1-\alpha) \mathbf{1} \cdot \bar x_k + \eta \mathbf{G_k} -\mathbf{y_k} \Vert \\
\quad & + \frac{1}{m} \Vert \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k +\eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k)- \eta \mathbf{G_k} \Vert  + \Vert \bar G_k - \tilde \nabla h(\bar y_k) \Vert \Vert \alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k \Vert \\ 
=& (1-\alpha)\mathcal{V_k} + \mathcal{I_1} +  \mathcal{I_2} + \mathcal{I_3} + \mathcal{I_4}
\end{align}
$$


对于与最优值 $x_{\ast}$ 相关的量以及真实广义梯度 $\tilde \nabla h(\bar y_k)$ 的范数，可以被 $\sqrt{\mathcal{V_k}}$ 进行控制，基于之前如下结论，


$$
\begin{align}
\Vert \bar y_k - x_{\ast} \Vert &\le  2 \sqrt{\frac{\mathcal{V_k}}{\mu}} \\
\Vert \bar x_k - x_{\ast} \Vert &\le  \sqrt{\frac{2\mathcal{V_k}}{\mu}} \le 2 \sqrt{\frac{\mathcal{V_k}}{\mu}}\\
\Vert \bar v_k - x_{\ast} \Vert &\le  \sqrt{\frac{2\mathcal{V_k}}{\mu}} \le 2 \sqrt{\frac{\mathcal{V_k}}{\mu}}\\
\end{align}
$$
进而可以得到，


$$
\begin{align}
\Vert \tilde \nabla h(\bar y_k) \Vert &= \Vert \tilde \nabla h(\bar y_k) - \tilde
\nabla h(x_{\ast})\Vert \\
&\le 3L \Vert \bar y_k - x_{\ast} \Vert \\
&\le 6L \sqrt{\frac{\mathcal{V_k}}{\mu}} \\
\Vert (1-\alpha) \bar x_k + \alpha x_{\ast} - \bar y_k \Vert &\le (1-\alpha) \Vert \bar x_k - \bar y_k \Vert + \alpha \Vert \bar y_k - x_{\ast}\Vert \\
&\le (1-\alpha) \Vert \bar x_k - x_{\ast} \Vert + (1-\alpha) \Vert \bar y_k - x_{\ast} \Vert + \alpha \Vert \bar y_k - x_{\ast} \Vert \\
&\le 2 \sqrt{\frac{\mathcal{V_k}}{\mu}}\\
\end{align}
$$


因此可以将上面关于Lyapunov函数的递推不等式转化为被 $\mathbf{z_k}, \sqrt{\mathcal{V_k}}$ 所控制，利用Gradient Tracking中的引理，



$$
\begin{align}
 \Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \bar G_k \Vert 
&\le 3 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert \\
\Vert \eta \mathbf{1} \cdot \bar G_k - \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert 
&\le3 \Vert \mathbf{y_{k}} - \mathbf{1} \cdot \bar y_k \Vert + \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert \\
\Vert \eta \mathbf{G_k} - \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert 
&\le 6 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 3 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert \\
\Vert \mathbf{s_k} - \nabla f(\mathbf{x_k}) \Vert 
&\le  \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert  + 2L \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert \\
\end{align}
$$



因此可以得到我们希望Bound的第一个量，



$$
\begin{align}
\mathcal{I_1} &=\frac{1}{2Lm} (\Vert \mathbf{1} \cdot \bar G_k \Vert^2-  \Vert \mathbf{G_k} \Vert^2) \\ &= \frac{1}{2Lm}(\Vert \mathbf{1} \cdot \bar G_k - \mathbf{1} \cdot \tilde \nabla h(\bar y_k ) \Vert^2 - \Vert \mathbf{G_k} -\mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert^2 +2 (\mathbf{1} \cdot\tilde \nabla h(\bar y_k))^T(\mathbf{1} \cdot \bar G_k - \mathbf{G_k})) \\
&\le \frac{1}{2Lm} \Vert \mathbf{1} \cdot \bar G_k - \mathbf{1} \cdot \tilde \nabla h(\bar y_k)\Vert^2 + \frac{1}{L \sqrt m} \Vert \tilde \nabla h(\bar y_k) \Vert \Vert \mathbf{G_k} - \mathbf{1} \cdot \bar G_k \Vert \\
&\le \frac{L}{m} (36 \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + 4 \eta^2 \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2)  + \frac{6}{\sqrt m}  \sqrt{\frac{\mathcal{V_k}}{\mu}}  (3 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 2\eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert) \\
&= \frac{36L}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + \frac{4}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2+ \frac{18}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{12}{L \sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$



类似地对于我们希望Bound的其他几个量，可以有


$$
\begin{align}

\mathcal{I_3} &=  \frac{1}{m} \Vert \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert \Vert \mathbf{y_k} - \mathbf{1} \cdot \bar y_k +\eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k)- \eta \mathbf{G_k} \Vert \\
&\le \frac{6L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} (\Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert +  \Vert \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k)- \eta \mathbf{G}_k \Vert) \\
&\le \frac{6L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}(7 \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 3 \eta \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert) \\
&= \frac{42 L}{\sqrt m }\sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{18}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert 
\end{align}
$$


以及，


$$
\begin{align}
\mathcal{I_4} &= \Vert \bar G_k - \tilde \nabla h(\bar y_k) \Vert \Vert \alpha x_{\ast} +(1-\alpha) \bar x_k -\bar y_k \Vert \\
&\le \frac{2}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} (3L \Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert +  \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert) \\
&= \frac{6L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{2}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$




最后只剩下比较长的一项，


$$
\begin{align}
\mathcal{I_2} &= \frac{1}{m}\Vert \mathbf{s_k} - \nabla f(\mathbf{y_k} ) + \mathbf{1} \cdot \tilde \nabla h(\bar y_k) - \mathbf{G_k} \Vert \Vert \alpha \mathbf{1} \cdot x_{\ast} + (1-\alpha) \mathbf{1} \cdot \bar x_k + \eta \mathbf{G_k} -\mathbf{y_k} \Vert \\
&= \mathcal{J1} \times \mathcal{J_2}
\end{align}
$$


我们分开计算，一方面，


$$
\begin{align}
\mathcal{J_1} &=\Vert \mathbf{s_k} - \nabla f(\mathbf{y_k} ) + \mathbf{1} \cdot \tilde \nabla h(\bar y_k) - \mathbf{G_k} \Vert \\
&\le \Vert \mathbf{s}_k - \nabla f(\mathbf{y}_k) \Vert + \Vert \mathbf{G}_k - \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert \\
&\le \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert  + 2L \Vert \mathbf{x_k} - \mathbf{1} \cdot \bar x_k \Vert + 6 L\Vert \mathbf {y_k} - \mathbf{1} \cdot \bar y_k \Vert + 3  \Vert \mathbf{s_k} - \mathbf{1} \cdot \bar s_k \Vert \\
&=6L \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + 2 L \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + 4 \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert
\end{align}
$$


另一方面，


$$
\begin{align}
\mathcal{J_2} &= \frac{1}{m} \Vert \alpha \mathbf{1} \cdot x_{\ast} + (1-\alpha) \mathbf{1} \cdot \bar x_k + \eta \mathbf{G_k} -\mathbf{y_k} \Vert  \\&\le  \Vert \alpha  x_{\ast} + (1-\alpha)  \bar x_k - \bar y_k \Vert + \frac{1}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{1}{m} \Vert \eta \mathbf{G}_k - \eta \mathbf{1} \cdot \tilde \nabla h(\bar y_k) \Vert + \eta \Vert \tilde \nabla h(\bar y_k) \Vert \\
&\le  \frac{2}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} + \frac{1}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert  +  \frac{6}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert+\frac{3}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert + \frac{6}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \\
&= \frac{8}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} + \frac{7}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert  +  \frac{3}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert  \\
\end{align}
$$

合起来就可以得到，



$$
\begin{align}
\mathcal{I_2} &\le \frac{48L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{16L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +\frac{32}{\sqrt m}  \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert  + \frac{42}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + \frac{12}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 \\
& \quad + \frac{14L}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{6}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert + \frac{46}{m}\Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert \\
&\le \frac{48L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{16L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +\frac{32}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert  \\
&\quad + \frac{42+30L}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + \frac{12+26L}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 + \frac{3+7L}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2
\end{align}
$$



将所有的 $\mathcal{I_1,I_2,I_3,I_4}$ 的Bound联合起来，



$$
\begin{align}
\mathcal{V_{k+1}} &\le (1-\alpha) \mathcal{V_k} + \mathcal{I_1} + \mathcal{I_2} + \mathcal{I_3} +\mathcal{I_4}   \\
&\le (1-\alpha) \mathcal{V_k}  \\
&\quad +\frac{36L}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + \frac{4}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2+ \frac{18}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{12}{L\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \\
&\quad +\frac{48L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{16L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +\frac{32}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}}  \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert  \\
&\quad + \frac{42+30L}{m} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert^2 + \frac{12+26L}{Lm} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 + \frac{3+7L}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 \\
&\quad  + \frac{42L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{18}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \\
&\quad + \frac{6L}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{y}_k - \mathbf{1} \cdot \bar y_k \Vert + \frac{2}{\sqrt m} \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \\
&\le (1-\alpha )\mathcal{V_k} + \frac{1}{m}(71+73L+ \frac{16}{L}) \Vert \mathbf{z_k} \Vert^2 + \frac{1}{\sqrt m} (70+ 112L + \frac{12}{L}) \sqrt{\frac{\mathcal{V_k}}{\mu}} \Vert \mathbf{z_k} \Vert 
\end{align}
$$



## Main Theorem for Convergence



之前的推导过程已经建立起了一个完整的不等式线性系统，基于此可以得到收敛性的证明，先令一些常数项


$$
\begin{align}
\mathcal{V_{K+1}} &\le (1-\alpha) \mathcal{V_k} + \frac{D_1}{\sqrt m} \sqrt{\mathcal{V_k}} \Vert \mathbf{z_k} \Vert + \frac{D_2}{m} \Vert \mathbf{z_k} \Vert^2 ,\Vert A \Vert_2 \le \rho D_3 \\ 
\text{With } D_1 &= (70+ 112L + \frac{12}{L}) \sqrt \mu ,D_2 = 71+73L + \frac{16}{L}, D_3 = 14+12L + \frac{6}{L} \\  
\end{align}
$$


假设 $\alpha > \frac{1}{2}$ , 对于 $\mathcal{V_1}$ 有


$$
\begin{align}
\mathcal{V_1} &\le (1-\alpha) \mathcal{V_0} + \frac{D_1}{\sqrt m} \sqrt{\mathcal{V_0}} \Vert \mathbf{z_0} \Vert + \frac{D_2}{m} \Vert \mathbf{z_0} \Vert^2 \\
&\le (1-\alpha) \mathcal{V_0} + \frac{\alpha}{2 }\mathcal{V_0} + \frac{D_1^2}{2\alpha m} \Vert \mathbf{z_0} \Vert^2 + \frac{D_2}{m} \Vert \mathbf{z_0} \Vert^2 \\
&\le (1-\frac{\alpha}{2}) \mathcal{V_0} + (1-\frac{\alpha}{2}) (\frac{D_1^2}{\alpha m} + \frac{2D_2}{m}) \Vert \mathbf{z_0} \Vert^2 \\
&\le(1-\frac{\alpha}{2}) (\mathcal{V_0} + \frac{C}{m}  \Vert \mathbf{z}_0  \Vert^2) \\
\text{Let } C &= D_1^2 \sqrt \kappa + 2D_2 
\end{align}
$$


令归纳假设为，


$$
\mathcal{V_{k}} \le (1-\frac{\alpha}{2})^k (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
$$


归纳下去可以得到，


$$
\begin{align}
 \mathbf{z}_{k+1}  &\le A^{k+1} \mathbf{z_0} + 8L \rho \sqrt m \sum_{i=0}^k A^{k-i} (0,0,\sqrt{\frac{\mathcal{V_i}}{\mu}})^T \\
\Vert \mathbf{z}_{k+1} \Vert &\le \rho D_3 2^{-k} \Vert \mathbf{z_0} \Vert+ 8L\rho \sqrt {\frac{m}{\mu}} \sum_{i=0}^k 2^{-(k-i)} \sqrt{\mathcal{V_i}}\\
&\le \rho D_3 2^{-k} \Vert \mathbf{z_0} \Vert + 8L \rho \sqrt {\frac{m}{\mu}} \sum_{i=0}^k 2^{-(k-i)} (\sqrt {1- \frac{\alpha}{2}})^i (\sqrt{\mathcal{V_0}} + \sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert)\\
&= \rho D_3 2^{-k} \Vert \mathbf{z_0} \Vert + 8L \rho \sqrt {\frac{m}{\mu}} \frac{2(\sqrt{1 -\frac{\alpha}{2} })^{k+1}-2^{-k}}{2 \sqrt{1 -\frac{\alpha}{2} }-1} (\sqrt{\mathcal{V_0}} +\sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert)\\
&\le \rho D_3 2^{-k} \Vert \mathbf{z_0} \Vert + 24L \rho \sqrt{\frac{m}{\mu}} (\sqrt{1-\frac{\alpha}{2}})^{k+1} (\sqrt{\mathcal{V_0}} + \sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert)\\ 
&\le 2\rho D_3  (\sqrt{1-\frac{\alpha}{2}})^{k+1} \Vert \mathbf{z}_0 \Vert +24L \rho \sqrt{\frac{m}{\mu}} (\sqrt{1-\frac{\alpha}{2}})^{k+1} (\sqrt{\mathcal{V_0}} + \sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert)\\ 
&=\rho (\sqrt{1-\frac{\alpha}{2}})^{k+1} D_4 \sqrt m(\sqrt{\mathcal{V_0}} + \sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert ) ,\text{Let } D_4 = 2D_3 \sqrt{\frac{1}{C}}  + 24L \sqrt{\frac{1}{\mu}}
\end{align}
$$


因而，


$$
\begin{align}
\mathcal{V_{k}} &\le (1-\frac{\alpha}{2})^k (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\ 
\Vert \mathbf{z}_k \Vert &\le \rho (\sqrt{1-\frac{\alpha}{2}})^{k} D_4 \sqrt m(\sqrt{\mathcal{V_0}} + \sqrt{\frac{C}{m}} \Vert \mathbf{z_0} \Vert ) \\
&\le \sqrt{2}\rho (\sqrt{1-\frac{\alpha}{2}})^{k} D_4 \sqrt m  \sqrt{\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2}\\
\Vert \mathbf{z}_k \Vert^2 &\le 2\rho^2 (1-\frac{\alpha}{2})^k D_4^2 m (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
&\le 2\rho (1-\frac{\alpha}{2})^k D_4^2 m(\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
\sqrt{\mathcal{V_k}} \Vert \mathbf{z_k} \Vert &\le  \sqrt 2 \rho (1-\frac{\alpha}{2})^k D_4 \sqrt m(\mathcal{V_0}+ \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
&\le 2 \rho (1-\frac{\alpha}{2})^k D_4 \sqrt m(\mathcal{V_0}+ \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
\end{align}
$$

从而，



$$
\begin{align}
\mathcal{V_{K+1}} &\le (1-\alpha) \mathcal{V_k} + \frac{D_1}{\sqrt m} \sqrt{\mathcal{V_k}} \Vert \mathbf{z_k} \Vert + \frac{D_2}{m} \Vert \mathbf{z_k} \Vert^2 \\
&\le (1-\alpha) (1-\frac{\alpha}{2})^k (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) + 2D_1D_4 \rho(1-\frac{\alpha}{2})^k (\mathcal{V_0}+ \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) 
+ 2D_2 D_4^2\rho  (1-\frac{\alpha}{2})^k  (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2)  \\
&\le (1-\frac{\alpha}{2})^{k+1} (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2)  + \rho D_5 (1-\frac{\alpha}{2})^{k}(\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2)  ,\text{Let } D_5 = 2 D_1 D_4+ 2D_2 D_4^2  \\
&\le (1-\frac{\alpha}{2})^{k+1}(\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) , \text{Let } \rho D_5 \le \frac{\alpha}{2}  \le 1- \frac{\alpha}{2} 
\end{align}
$$



且 $D_1,D_2,D_3,D_4,D_5$ 都是与机器数目 $m$ 无关的量，最终完成归纳，


$$
\mathcal{V_{k}} \le (1-\frac{\alpha}{2})^k (\mathcal{V_0} + \frac{C}{m} \Vert \mathbf{z_0} \Vert^2) \\
$$


忽略 $\log$ 项， 达到 $\epsilon$ -最优解的计算复杂度为 $\tilde O(\sqrt \kappa \log \frac{1}{\epsilon})$,  

考虑到FastMix算子的通信，通信复杂度为 $\tilde O(\sqrt{\frac{\kappa}{1- \lambda_2(W)}})  \log \frac{1}{\epsilon}$
