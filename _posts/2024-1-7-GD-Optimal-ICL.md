---
title: 'GD is Optimal for In-Context Learning'
toc: true
excerpt_separator: <!--more-->
tags: 		
  - Transformer理论

---



Paper Reading: One Step of Gradient Descent is Provably the Optimal In-Context Learner with One Layer of Linear Self-Attention.

<!--more-->



近期工作 [4] 证明了单层的线性Attention网络可以使用简洁的构造实现梯度下降算法(GD), 本次介绍的工作 [1] 证明了这个构造是in-context loss的最优点，也即最小化in-context loss的最优网络就是在实现梯度下降。同时期的工作 [2,3] 也证明了类似的结果，然而 [1] 中的证明似乎更为简洁且容易推广，因此本文着重介绍 [1].



## Problem Setup



类似于 [4] 的问题设定，考虑单层线性Attention的Transformer对于线性回归问题。

考虑标准的线性回归模型，数据分布 $x_i \sim \mathcal{N}(0_d,I_{d \times d})$, $w \sim \mathcal{N}(0_d,I_{d \times d})$,  $\epsilon_i \sim \mathcal{N}(0,\sigma^2), $$y_i = w^\top x_i + \epsilon_i$.

给定大小为 $n$ 的数据集 $D$, 样本为 $(x_1,y_1), \cdots (x_n,y_n)$. 以及测试点 $x_{n+1} $.

我们使用一个单层线性Attention的Transformer进行预测，其中输入的Embedding为 $h_i = (x_i; y_i)$, $h_{n+1} = (x_{n+1};0)$.

模型的输出为最后一个embedding的最后一维，注意到Attention机制不涉及自己，



$$
\begin{align*}
\hat y_{n+1} = e^\top  \left( \sum_{n=1}^n (W_V h_i) (W_K h_i)^\top W_Q h_{n+1} \right).
\end{align*}
$$

定义记号



$$
\begin{align*}
G_D &= \sum_{i=1}^n \begin{pmatrix} x_i \\ y_i\end{pmatrix} \begin{pmatrix} x_i \\ y_i\end{pmatrix}^\top = \sum_{i=1}^n \begin{pmatrix}  x_i x_i^\top & y_i x_i \\
y_i x_i^\top & y_i^2 \end{pmatrix} \\
X &= 
\begin{pmatrix} 
x_1^\top \\
\vdots \\
x_n^\top \\
\end{pmatrix}, \quad \vec{y} = 
\begin{pmatrix}
y_1 \\
\vdots \\
y_n
\end{pmatrix}, \quad w =  W_V^\top h, \quad
M = W_K^\top W_Q.
\end{align*}
$$



我们希望最小化如下的In-Context Loss，用上面的记号可以给出如下的等价形式.



$$
\begin{align*}
\min_{W_V, W_K, W_Q} L(W_V,W_Q,W_V,e)  &:=  \mathbb{E}_{D , x_{n+1}, y_{n+1}} \left[ (\hat y_{n+1} - y_{n+1})^2 \right] \\
\min_{w,M} L(w,M) &:= \mathbb{E}_{D , x_{n+1}, y_{n+1}}\left[ (w^\top G_D M h_{n+1} - y_{n+1})^2 \right].
\end{align*}
$$




## Main Result and Proof



文章证明了最小化上述的In-Context Loss可以在如下的最优参数下取到，其中



$$
\begin{align*}
e^\ast = \begin{pmatrix}
0_d \\
1 
\end{pmatrix}, \quad
W_V^\ast = \begin{pmatrix}
0_{d \times d} & 0_d \\
0_d^\top & \eta 
\end{pmatrix}, \quad
W_K^\ast = W_Q^\ast =  \begin{pmatrix}
I_{d \times d} & 0_d \\
0_d^\top & 0 
\end{pmatrix}.
\end{align*}
$$



上述的最优参数使得模型的输出满足



$$
\begin{align*}
\hat y_{n+1} &= 
\begin{pmatrix}
0_{d \times d} &\eta
\end{pmatrix}
\sum_{i=1}^n \begin{pmatrix} x_i \\ y_i\end{pmatrix} \begin{pmatrix} x_i \\ y_i\end{pmatrix}^\top \begin{pmatrix}
I_{d \times d} & 0_d \\
0_d^\top & 0 
\end{pmatrix} 
\begin{pmatrix}
x_{n+1} \\
0 
\end{pmatrix} \\
&=  \eta \sum_{i=1}^n y_i x_i^\top x_{n+1}.
\end{align*}
$$



这正好等价于在如下的最小二乘损失上面进行了一步GD步之后的参数在测试点 $x_{n+1}$ 的预测值



$$
\begin{align*}
L(w) = \sum_{i=1}^n (w^\top x_i - y_i)^2.
\end{align*}
$$



下面我们证明上述定理。首先我们定义给定先验 $D$ 时最优的后验预测器



$$
\begin{align*}
\hat w_D &= \arg \min_u \mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ ( u^\top x_{n+1} - y_{n+1})^2  \right] \\
&= \arg \min_u \mathbb{E}_{w \mid D} \mathbb{E}_{x_{n+1}}  \left[ \left( (u - w)^\top x_{n+1} \right)^2 + \sigma^2 \right] \\
&=  \arg \min_u \mathbb{E}_{w \mid D} \mathbb{E}_{x_{n+1}}   \left[ \left( (u - w)^\top x_{n+1} \right)^2\right] \\
&= \arg \min_u \mathbb{E}_{w \mid D} \Vert u - w \Vert^2 \\
&= \mathbb{E} [w \mid D].
\end{align*}
$$



可以验证后验分布 $p(w \mid D)$ 具有如下的形式，



$$
\begin{align*}
p(w \mid D) &\propto p(w) p( X, \vec{y} \mid w) \\
&\propto \exp\left( - \frac{1}{2} w^\top w\right) \exp \left( - \frac{1}{2 \sigma^2} (\vec y - X w )^\top (\vec y - X w) \right) \\
&= \mathcal{N}_d ( \hat w_{\rm ridge}, \Sigma ) 
\end{align*}
$$



其中均值为下面岭回归的解，



$$
\begin{align*}
\hat w_{\rm ridge} = \arg \min_{w}  \frac{1}{2} \Vert \vec y - Xw \Vert^2 + \frac{\sigma^2}{2} \Vert w \Vert^2.
\end{align*}
$$



因此我们知道 $\hat w_D = \mathbb{E}[ w \mid D] = \hat w_{\rm ridge}$. 根据一阶最优性条件，我们有



$$
\begin{align*}
\mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ (\hat w_D^\top x_{n+1} - y_{n+1}) x_{n+1} \right] = 0.
\end{align*}
$$



由于 $v_{n+1} = (x_{n+1}; 0)$ ，其最后一个分量为 $0$， 我们如下对损失函数进行化简



$$
\begin{align*}
 L(w,M) &= \mathbb{E}_D\mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ (w^\top G_D M h_{n+1} - y_{n+1})^2 \right] \\
&=\mathbb{E}_D\mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ (w^\top G_D M_{:,1:d} x_{n+1} - y_{n+1})^2 \right] \\
&= \mathbb{E}_D\mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ w^\top G_D M_{:,1:d} x_{n+1} - \hat w_D^\top x_{n+1} + \hat w_D^\top x_{n+1} - y_{n+1})^2 \right] \\
&= \mathbb{E}_D \mathbb{E}_{x_{n+1}}\left[ (w^\top G_D M_{:,1:d,} x_{n+1} - \hat w_D^\top x_{n+1})^2 \right]  +  \mathbb{E}_D\mathbb{E}_{x_{n+1},y_{n+1} \mid D} \left[ ( \hat w_D^\top x_{n+1} - y_{n+1})^2 \right]  \\
&= \mathbb{E}_D \left \Vert A G_D  w - \hat w_D  \right \Vert^2 +C ,\quad A =M_{:,1:d}^\top.
\end{align*}
$$



其中倒数第二行用到了 $\hat w_D$ 的一阶最优性条件，使得倒数第三行的二次项展开后的交叉项为 $0$. 此即 [1] 中的Lemma 1.

下面我们沿着上面的式子继续进行证明，我们首先证明 [1] 中的Lemma 3 和 4, 阐述如下。

由于分布 $x,w,y$ 均为球面对称分布，可以验证存在常数 $c_1,c_2$ 使得



$$
\begin{align*}
\mathbb{E}_D \left[X^\top \vec y \vec y^\top X \right] = c_1 I_{d \times d} ,\quad \mathbb{E}_D \left[ 
X^\top \vec y  ~ \hat w_D^\top\right] = c_2 I_{d \times d}.
\end{align*}
$$



这就说明了存在 $\eta$ 使得



$$
\begin{align*}
\mathbb{E}_D \left[ 
X^\top \vec y  ~ \hat w_D^\top\right] = \eta \mathbb{E}_D \left[X^\top \vec y \vec y^\top X \right].
\end{align*}
$$



根据上述引理，我们下面证明 [1] 中的 Lemma 2, 也即存在常数 $C$ 使得



$$
\begin{align*}
\mathbb{E}_D \left \Vert A G_D  w - \hat w_D  \right \Vert^2 = \mathbb{E}_D \Vert A G_D w - \eta X^\top \vec y \Vert^2 +C,\quad \forall A,w.
\end{align*}
$$



我们只需要证明上面的两个函数关于 $A,w$ 的梯度都相等。注意到形式上的相似性，我们下面仅给出关于 $w$ 的证明，而关于 $A$ 的证明是完全类似的，细节可以参考 [1] 中的详细推导。考虑上面两个函数关于 $w$ 的导数，我们希望有


$$
\begin{align*}
&\quad \mathbb{E}_D [ G_D A^\top (A G_D w - \hat w_D) ]=  \mathbb{E}_D  [ G_DA^\top (AG_D w - \eta X^\top \vec y) ] \\
& \Leftrightarrow \mathbb{E}_D [G_DA^\top \hat w_D ] = \eta \mathbb{E}_D [ G_DA^\top X^\top \vec y ] \\
&  \Leftrightarrow \sum_{i=1}^n \mathbb{E} \begin{bmatrix}
x_i x_i^\top A^\top_{:,1:d} \hat w_D & y_i x_i  A^\top_{:,d+1} \hat w_D \\
y_i x_i^\top A_{:,1:d}^\top \hat w_D & y_i^2 A_{:,d+1}^\top \hat w_D 
\end{bmatrix} =
\eta\sum_{i=1}^n \mathbb{E} \begin{bmatrix}
x_i x_i^\top A^\top_{:,1:d}X^\top \vec y  & y_i x_i  A^\top_{:,d+1} X^\top \vec y \\
y_i x_i^\top A_{:,1:d}^\top X^\top \vec y  & y_i^2 A_{:,d+1}^\top X^\top \vec y 
\end{bmatrix} \\
&{\Leftrightarrow}  \sum_{i=1}^n \mathbb{E} \begin{bmatrix}
0_{d \times d} & y_i x_i  A^\top_{:,d+1} \hat w_D \\
y_i x_i^\top A_{:,1:d}^\top \hat w_D & 0 
\end{bmatrix} =
\eta\sum_{i=1}^n \mathbb{E} \begin{bmatrix}
0_{d \times d}  & y_i x_i  A^\top_{:,d+1} X^\top \vec y \\
y_i x_i^\top A_{:,1:d}^\top X^\top \vec y  & 0
\end{bmatrix}, \quad (\text{by symmetry}) \\
& \Leftrightarrow    \mathbb{E} \begin{bmatrix}
0_{d \times d} & X^\top \vec y   A^\top_{:,d+1} \hat w_D \\
\vec y^\top X A_{:,1:d}^\top \hat w_D & 0 
\end{bmatrix} = \eta \mathbb{E} \begin{bmatrix}
0_{d \times d} & X^\top \vec y   A^\top_{:,d+1} X^\top \vec y \\
\vec y^\top X A_{:,1:d}^\top X^\top \vec y & 0 
\end{bmatrix} \\
&\Leftrightarrow    \mathbb{E} \begin{bmatrix}
0_{d \times d} & X^\top \vec y   \hat w_D^\top A_{:,d+1}  \\
{\rm tr} \left( A_{:,1:d}^\top \hat w_D \vec y^\top X \right) & 0 
\end{bmatrix} = \eta \mathbb{E} \begin{bmatrix}
0_{d \times d} & X^\top \vec y  \vec y^\top X A_{:,d+1}  \\
{\rm tr} \left( A_{:,1:d}^\top X^\top \vec y \vec y^\top X \right)& 0 
\end{bmatrix}
\end{align*}
$$


根据之前的引理我们知道的根据分布的球面对称性，确存在常数 $\eta$ 使得上式左右两边相等。

至此，我们经过一番计算成功地证明了


$$
\begin{align*}
L(w,M) &=  \mathbb{E}_D \left \Vert A G_D  w - \hat w_D  \right \Vert^2 + {\rm const} ,\quad A =M_{:,1:d}^\top \\
&= \mathbb{E}_D \left \Vert A G_D  w - \eta X^\top \vec y  \right \Vert^2 + {\rm const}.
\end{align*}
$$


注意到对于给定的可以实现GD的参数，有


$$
\begin{align*}
A^\ast G_D^\ast w^\ast  = \begin{pmatrix}
I_{d \times d}& 0 
\end{pmatrix}
\sum_{i=1}^n \begin{pmatrix}  x_i x_i^\top & y_i x_i \\
y_i x_i^\top & y_i^2 \end{pmatrix} 
\begin{pmatrix}
0_{d \times d} \\
\eta
\end{pmatrix}
= \eta X^\top \vec y. 
\end{align*}
$$


这正好使得损失函数的第一项为 $0$, 恰为该损失函数的最优点。



## Different Data Covariance Matrices



文章的结论很容易推广至更general的协方差情形，考虑数据分布 


$$
\begin{align*}
x_i \sim \mathcal{N} (0_d, \Sigma), \quad w \sim \mathcal{N}(0_d, \Sigma^{-1}), \quad \epsilon_i \sim \mathcal{N}(0,\sigma^2).
\end{align*}
$$


此时只要经过换元就可以使用上一章节的结论，此时最优的Transformer参数在实现一个预处理的梯度下降, 产生预测，


$$
\begin{align*}
\hat y_{n+1} &= \eta \sum_{i=1}^n y_i( \Sigma^{-1}x_i)^\top x_{n+1}.
\end{align*}
$$




## Nonlinear Target Functions



有趣的是，文章的大多数证明仅仅依赖于球面对称性，因此文章的结论很容易推广到GroundTrue模型为非线性模型的情况，


$$
\begin{align*}
y_i = f(x_i) + \epsilon_i.
\end{align*}
$$


只要该函数 $f$ 从满足如下假设的分布上采样，GD仍然是In-Context Loss的最优点：

此即 [1] 中的Assumption 1，具体来说，需要 满足旋转不变性以及对称性，即


$$
\begin{align*}
p(f) \sim p(R f), \quad p(f) = p(-f),
\end{align*}
$$


其中 $R$ 为任意的旋转矩阵。此时最优的Transformer参数仍然产生预测


$$
\begin{align*}
\hat y_{n+1} &= \eta \sum_{i=1}^n y_i x_i^\top x_{n+1}.
\end{align*}
$$


## Reference

[1] Mahankali, Arvind, Tatsunori B. Hashimoto, and Tengyu Ma. "One step of gradient descent is provably the optimal in-context learner with one layer of linear self-attention." arXiv preprint, 2023.

[2] Ahn, Kwangjun, et al. "Transformers learn to implement preconditioned gradient descent for in-context learning." In NeurIPS,2023.

[3] Zhang, Ruiqi, Spencer Frei, and Peter L. Bartlett. "Trained Transformers Learn Linear Models In-Context." arXiv preprint, 2023.

[4] Von Oswald, Johannes, et al. "Transformers learn in-context by gradient descent." In ICML, 2023.
