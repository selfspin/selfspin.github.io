---
title: 'Benign Overfitting of SGD for Least Square'
toc: true
excerpt_separator: <!--moer-->
tags:
  - 随机优化
---



Paper Reading: Benign Overfifitting of Constant-Stepsize SGD for Linear Regression (JMLR' 21 and COLT' 21)



<!--more-->



文章可以看作 [Least Square SGD with Tail Average](https://truenobility303.github.io/SGD-LS/) 的更精细的界，给出了在高维情况下几乎匹配的上界和下界。



## Introdcution and Assumption 



考虑使用SGD (随机梯度下降) 训练以下线性回归问题，





$$
\begin{align*}
\min_w L(w) = \frac{1}{2} \mathbb{E}_{(x,y) \in \mathcal{D} } [ (y -w^\top x)^2].
\end{align*}
$$





假设 $H = \mathbb{E}[xx^\top]$ 存在并且严格正定，且对于其四阶矩满足存在正常数使得，





$$
\begin{align*}
\mathbb{E}[xx^\top Ax x^\top] \preceq \alpha  {\rm tr} (HA) H, \quad \forall A
\end{align*}
$$





可以验证该四阶矩条件弱于一般线性回归模型下的正态噪声假设的条件，并且定义如下的噪声量，





$$
\begin{align*}
\Sigma = \mathbb{E}[ (y - w_*^\top x)^2 xx^\top], \quad \sigma^2 = \Vert H^{-1/2} \Sigma H^{-1/2} \Vert.
\end{align*}
$$





注意到 $\Sigma$ 为最优点处梯度的协方差， $\sigma^2$ 则推广了一般线性回归模型下正态噪声的方差量 $\sigma^2$。



## Risk Upper Bound



证明的框架遵循  [Least Square SGD with Tail Average](https://truenobility303.github.io/SGD-LS/)， 但本文为了处理高维甚至无穷维的情况，需要更精细的控制。



首先根据偏差-方差分解，可以证明，





$$
\begin{align*}
L(w_t) - L(w_*) \le  \mathbb{E}[ \eta_{t, {\rm bias}}^\top H \eta_{t, {\rm bias}} ] + \mathbb{E} [\eta_{t, {\rm variance}}^\top H \eta_{t, {\rm variance}} ]
\end{align*}
$$





其中对应的量 $\eta$ 分别由下面两个迭代序列产生，





$$
\begin{align*}
\eta_{t, {\rm bias}} &=  (I - \gamma x_t x_t^\top) \eta_{t-1,{\rm bias}}, \quad \eta_{0, {\rm bias}} = w_0 - w_*  \\
\eta_{t, {\rm variance}} &= (I - \gamma x_t x_t^\top ) \eta_{t-1, {\rm variance}} + \gamma \xi_t, \quad \eta_{0, {\rm variance}} = 0
\end{align*}
$$





其中 $\xi_t$ 为期望为0的噪声项： $\xi_t = x_t (w_*^\top x_t - y_t)$.



因此在期望意义下，成立如下的递推，



$$
\begin{align*}
\mathbb{E}[ \eta_{t, {\rm bias}}] &= (I - \gamma H) \mathbb{E}[\eta_{t-1, {\rm bias}}] \\
\mathbb{E}[ \eta_{t, {\rm variance}}] &= (I - \gamma H) \mathbb{E}[\eta_{t-1, {\rm variance}}].
\end{align*}
$$



定义如下的协方差矩阵，





$$
\begin{align*}
B_t = \mathbb{E}[\eta_{t, {\rm bias}}\eta_{t, {\rm bias}}^\top] ,\quad C_t = \mathbb{E}[\eta_{t, {\rm variance}}\eta_{t, {\rm variance}}^\top] .
\end{align*}
$$



则可以得到对应的递推关系式为，



$$
\begin{align*}
B_{t} &= \mathbb{E}[ (I - \gamma x_t x_t) B_{t-1} (I - \gamma x_t x_t^\top)] \\
C_{t} &= \mathbb{E}[ (I - \gamma x_t x_t) C_{t-1} (I - \gamma x_t x_t^\top)] + \gamma^2 \Sigma.
\end{align*}
$$



定义如下算子，



$$
\begin{align*}
\mathcal{S} \circ M &= \mathbb{E}[xx^\top Mxx^\top], \quad \tilde {\mathcal{S}} \circ M = \mathbb{E}[HMH] \\
\mathcal{T} \circ M &= HM + MH - \gamma \mathcal{S} \circ M , \quad \tilde{\mathcal{T}} \circ M  = HM + MH - \gamma \tilde{\mathcal{S}} \circ M.
\end{align*}
$$





在上述定义下，

$$
\begin{align*}
(\mathcal{S}  - \tilde{\mathcal{S}}) \circ M  &=  \mathbb{E}[(xx^\top - H) M (xx^\top -H)] \\
(\mathcal{I} - \gamma \mathcal{T}) \circ M &= \mathbb{E}[ (I - \gamma xx^\top) M (I - \gamma xx^\top)] \\
(\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ M &= \mathbb{E}[ (I - \gamma H) M (I - \gamma H)]
\end{align*}
$$



并且根据Neumann级数， $\mathcal{T}^{-1}, \tilde{\mathcal{T}}^{-1}$ 都存在。



### Variance Bound



首先，证明 $C_t$ 单调递增。



$$
\begin{align*}
C_t &= (\mathcal{I} - \gamma \mathcal{T}) \circ C_{t-1} + \gamma^2 \Sigma \\
&= \gamma^2 \sum_{k=0}^{t-1} (\mathcal{I} - \gamma \mathcal{T})^k \circ \Sigma \\
&= C_{t-1} + \gamma^2 (\mathcal{I} - \gamma  \mathcal{T})^{t-1} \circ \Sigma \\
&\succeq C_{t-1}.
\end{align*}
$$



其次，证明 $C_t$ 存在上界，定义



$$
\begin{align*}
A_t = (\mathcal{I} - \gamma \mathcal{T})^t \circ \Sigma
\end{align*}
$$



可以得到递推关系式，当 $\gamma \le 1/ (\alpha {\rm tr}(H))$ 时，





$$
\begin{align*}
{\rm tr}(A_t) &= {\rm tr} (A_{t-1}) - 2 \gamma {\rm tr} ( A_{t-1} H) + \gamma^2 {\rm tr} (A_{t-1} \mathbb{E}[ xx^\top xx^\top]) \\
&\le {\rm tr} (A_{t-1}) - 2 \gamma {\rm tr} ( A_{t-1} H) + \gamma^2 \alpha {\rm tr}(H) {\rm tr} (A_{t-1} H) \\
&\le {\rm tr} (A_{t-1}) -  \gamma {\rm tr} ( A_{t-1} H) \\
&\le (1-  \gamma \mu ) {\rm tr} (A_{t-1}).
\end{align*}
$$





因此，



$$
\begin{align*}
{\rm tr }(C_t) \le {\rm tr}(C_{\infty}) \le \gamma^2 \sum_{k=0}^{\infty} (1- \gamma \mu)^k {\rm tr}(\Sigma) = \frac{\gamma}{\mu} {\rm tr} (\Sigma).
\end{align*}
$$



根据单调序列收敛定理, $C_{\infty}$ 存在并且满足，





$$
\begin{align*}
C_{\infty} &= \gamma \tilde {\mathcal{T}}^{-1} \circ (\mathcal{S} - \tilde{\mathcal{S}}^{-1}) \circ C_{\infty} +   \gamma \tilde {\mathcal{T}}^{-1} \circ \Sigma \\
&\preceq \gamma  \tilde {\mathcal{T}}^{-1} \circ \mathcal{S}  \circ C_{\infty} +   \gamma \tilde {\mathcal{T}}^{-1} \circ \Sigma \\
&\preceq \gamma \sum_{k=0}^{t-1} (\tilde{\gamma \mathcal{T}}^{-1} \circ \mathcal{S})^t \circ \tilde{\mathcal{T}}^{-1} \circ \Sigma \\
&\preceq \gamma \sigma^2 \sum_{k=0}^{\infty} (\tilde{\gamma \mathcal{T}}^{-1} \circ \mathcal{S})^t \circ \tilde{\mathcal{T}}^{-1} \circ H.
\end{align*}
$$





利用关系式





$$
\begin{align*}
\tilde{\mathcal{T}}^{-1} \circ H \preceq I, \quad \mathcal{S} \circ I \preceq \alpha {\rm tr}(H) H,
\end{align*}
$$

 



可以得到



$$
\begin{align*}
(\gamma \tilde{T}^{-1} \circ \mathcal{S})^t \circ \tilde{\mathcal{T}}^{-1} \circ H  \preceq (\gamma \alpha {\rm tr}(H))^t I.
\end{align*}
$$



因此有，



$$
\begin{align*}
C_{\infty} \preceq  \gamma \sigma^2 \sum_{k=0}^{\infty} (\gamma \alpha {\rm tr}(H))^k I = \frac{\gamma \sigma^2}{1- \gamma \alpha {\rm tr}(H)} I.
\end{align*}
$$



巧妙的是使用在上述得到的较松的上界的基础上可以得到更紧的界，



$$
\begin{align*}
C_{t} &= (\mathcal{I} - \gamma \mathcal{T}) \circ C_{t-1} + \gamma^2 \Sigma \\
&= (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \gamma^2 (\mathcal{S} - \tilde{\mathcal{S}} ) \circ C_{t-1} +\gamma^2 \Sigma \\
&\preceq (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \gamma^2 \mathcal{S}  \circ C_{t-1} +\gamma^2 \Sigma \\
&\preceq (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \frac{\gamma^3 \sigma^2}{1- \gamma \alpha {\rm tr}(H)} \mathcal{S} \circ I
 + \gamma^2 \Sigma \\
&\preceq (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \frac{\gamma^3 \sigma^2 \alpha {\rm tr}(H)}{1- \gamma \alpha {\rm tr }(H)} H
 + \gamma^2 \Sigma \\
& \preceq (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \frac{\gamma^3 \sigma^2 \alpha {\rm tr}(H)}{1- \gamma \alpha {\rm tr }(H)} H
 + \gamma^2 \sigma^2 H \\ 
 &= (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ C_{t-1} + \frac{\gamma^2 \sigma^2}{1 - \gamma \alpha {\rm tr}(H)} H \\
 &\preceq \frac{\gamma^2 \sigma^2}{1 - \gamma \alpha {\rm tr}(H)} \sum_{k=0}^{t-1} (\mathcal{I} - \gamma \tilde{\mathcal{T}})^k \circ H \\
&= \frac{\gamma^2 \sigma^2}{1 - \gamma \alpha {\rm tr}(H)} \sum_{k=0}^{t-1} (I - \gamma H)^k H ( I - \gamma H)^k \\
&\preceq \frac{\gamma^2 \sigma^2}{1 - \gamma \alpha {\rm tr}(H)} \sum_{k=0}^{t-1} (I - \gamma H)^k H \\
&= \frac{\gamma \sigma^2}{1- \gamma \alpha{\rm tr}(H)} (I - (I - \gamma H)^t).
\end{align*}
$$



算法采用均值作为输出，对应的方差项为，



$$
\begin{align*}
\bar C_T &= \mathbb{E}[ (\bar w_T  - w_*) (\bar w_T - w_*)^\top  ] \\
&\preceq \frac{1}{T^2} \sum_{t = 0}^{T-1} \sum_{k =t}^{T-1} \mathbb{E} [ (w_t - w_*) (w_{k} - w_*)^\top + (w_{k} - w_*) (w_{t} - w_*)^\top] \\
&= \frac{1}{T^2} \sum_{t = 0}^{T-1} \sum_{k =t}^{T-1} \mathbb{E}[C_t ( I - \gamma H)^{k-t} +  ( I - \gamma H)^{k-t} C_t].
\end{align*}
$$



据此可以得到方差项的上界为，



$$
\begin{align*}
{\rm variance} &= \frac{1}{2}\mathbb{E}_{w_0 = w_*}[ (\bar w_T - w_*)^\top  H  (\bar w_T  - w_*) ] \\
&\le \frac{1}{T^2}  \sum_{t = 0}^{T-1} \sum_{k =t}^{T-1} {\rm tr} (C_t (I - \gamma H)^{k-t} H) \\
&= \frac{1}{\gamma T^2}  \sum_{t = 0}^{T-1}  {\rm tr} (C_t (I - (I- \gamma H)^{T-t})) \\
&\le \frac{\sigma^2}{T^2(1 - \gamma \alpha {\rm tr}(H))} \sum_{t = 0}^{T-1} {\rm tr} ((I - (I - \gamma H)^t)  (I - (I- \gamma H)^{T-t})) \\
&\le \frac{\sigma^2}{T^2(1 - \gamma \alpha {\rm tr}(H))} \sum_{i} \sum_{t=0}^{T-1} (1 - (1 -\gamma \lambda_i)^T)^2 \\
&\le \frac{\sigma^2}{T(1 - \gamma \alpha {\rm tr}(H))} \sum_{i}  (1 - (1 -\gamma \lambda_i)^T)^2 \\
&\le \frac{\sigma^2}{T(1 - \gamma \alpha {\rm tr}(H))} \sum_{i} \min\left\{ 1, T^2 \gamma^2 \lambda_i^2\right\} \\
&\le \frac{\sigma^2}{1 - \gamma \alpha {\rm tr}(H)} \left( \frac{k^*}{T} +  \gamma^2 T\sum_{i \ge k^*} \lambda_i^2\right), \quad k^* = \max\{k:\lambda_k  \ge 1/(\gamma T) \}.
\end{align*}
$$



其中 $k^*$ 可以理解为协方差矩阵 $H$ 的有效维数。







### Bias Bound





对于偏差项，定义 $S_t = \sum_{k=0}^{t-1} B_t$ , 可以得到如下的递推关系式，


$$
\begin{align*}
S_t = (\mathcal{I} - \gamma \mathcal{T}) \circ S_{t-1} + B_0.
\end{align*}
$$


以及成立，


$$
\begin{align*}
B_0 = S_0 \preceq S_1 \preceq \cdots \preceq S_t.
\end{align*}
$$




利用之前所定义的算子，成立如下关系，


$$
\begin{align*}
&\quad \mathcal{S} \circ \mathcal{T}^{-1} \circ M \\
&= \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M + \mathcal{S} \circ   ( \mathcal{T}^{-1} - \tilde{\mathcal{T}}^{-1}) \circ M  \\
&= \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M + \mathcal{S} \circ  \tilde{\mathcal{T}}^{-1} \circ( \tilde{\mathcal{T}} - \mathcal{T} ) \circ {\mathcal{T}}^{-1} \circ M \\
&= \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M + \gamma \mathcal{S} \circ  \tilde{\mathcal{T}}^{-1} \circ(  \mathcal{S} -\tilde{\mathcal{S}}) \circ {\mathcal{T}}^{-1} \circ M \\
&\preceq \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M + \gamma \mathcal{S} \circ  \tilde{\mathcal{T}}^{-1}  \circ \mathcal{S} \circ {\mathcal{T}}^{-1} \circ M \\
&\preceq \sum_{k=0}^{t-1} (\gamma \mathcal{S} \circ \tilde{\mathcal{T}}^{-1})^k \circ \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M. 
\end{align*}
$$


根据定义和假设，可以发现成立


$$
\begin{align*}
\mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M \preceq \alpha {\rm tr} (H \tilde{\mathcal{T}}^{-1} \circ M) H \preceq  \alpha {\rm tr} (M) H.
\end{align*}
$$






可以得到



$$
\begin{align*}
(\gamma  \mathcal{S} \circ \tilde{T}^{-1} )^t \circ \mathcal{S} \circ \tilde{\mathcal{T}}^{-1} \circ M  \preceq (\gamma \alpha {\rm tr}(M))^t  \alpha {\rm tr}(M) H.
\end{align*}
$$



因此有，



$$
\begin{align*}
\mathcal{S} \circ \mathcal{T}^{-1} \circ M  \preceq  \alpha {\rm tr}(M) \sum_{k=0}^{\infty} (\gamma \alpha {\rm tr}(H))^k H = \frac{\alpha {\rm tr}(M)}{1- \gamma \alpha {\rm tr}(H)} H.
\end{align*}
$$



进一步得到，


$$
\begin{align*}
\mathcal{S} \circ S_t &= \sum_{k=0}^{t-1} \mathcal{S} \circ(\mathcal{I} - \gamma \mathcal{T})^k \circ B_0 \\
&= \gamma \mathcal{S} \circ \mathcal{T}^{-1} (\mathcal{I} -(\mathcal{I} - \gamma \mathcal{T})^t ) \circ B_0 \\
&\preceq \gamma \mathcal{S} \circ \mathcal{T}^{-1} (\mathcal{I} -(\mathcal{I} - \gamma \tilde{\mathcal{T}})^t ) \circ B_0 \\
& \preceq \frac{\gamma \alpha H }{1- \gamma \alpha {\rm tr}(H)} {\rm tr} \left( (\mathcal{I} -(\mathcal{I} - \gamma \tilde{\mathcal{T}})^t ) \circ B_0  \right)
\end{align*}
$$




类似对于方差项的控制，上述的界可以用来得到精细的界，


$$
\begin{align*}
\mathcal{S}_t &\preceq (\mathcal{I} - \gamma \tilde{\mathcal{T}}) \circ S_{t-1} + \gamma^2 \mathcal{S} \circ S_t + B_0 \\
&\preceq  \sum_{k=0}^{t-1} (\mathcal{I} - \gamma \tilde{\mathcal{T}})^k \circ \left(\frac{\gamma \alpha H }{1- \gamma \alpha {\rm tr}(H)} {\rm tr} \left( (\mathcal{I} -(\mathcal{I} - \gamma \tilde{\mathcal{T}})^t ) \circ B_0  \right) + B_0  \right) \\
&= \sum_{k=0}^{t-1} ( I - \gamma H)^k \left( \frac{\gamma \alpha H }{1- \gamma \alpha {\rm tr}(H)} {\rm tr} (A_t) + B_0 \right) (I - \gamma H)^k.
\end{align*}
$$


代入偏差项



$$
\begin{align*}
{\rm bias} &= \frac{1}{2}\mathbb{E}_{\xi_t = 0}[ (\bar w_T - w_*)^\top  H  (\bar w_T  - w_*) ] \\
&\le \frac{1}{T^2}  \sum_{t = 0}^{T-1} \sum_{k =t}^{T-1} {\rm tr} (B_t (I - \gamma H)^{k-t} H) \\
&\le \frac{1}{\gamma T^2}  \sum_{t = 0}^{T-1}  {\rm tr} (B_t (I - (I- \gamma H)^{T})) \\
&= \frac{1}{\gamma T^2} {\rm tr} (S_t (I - (I- \gamma H)^{T})) \\
&\le \frac{1}{\gamma T^2}\sum_{k=0}^{t-1} {\rm tr} \left( \frac{\gamma \alpha  {\rm tr} (A_t) }{1- \gamma \alpha {\rm tr}(H)}  (I-\gamma H)^{2k} H +  (I - \gamma H)^{2k+T} B_0 \right).
\end{align*}
$$





对 $H$ 进行特征值分解并且控制 ${\rm tr}(A_t)$ 之后可以得到对应的偏差项的上界为，


$$
\begin{align*}
{\rm bias} &\le \frac{1}{\gamma^2 T^2} \Vert w_0 - w_* \Vert_{H_{0:k^*}^{-1}} + \Vert w_0 - w_* \Vert_{H_{k^*: \infty}}^2 \\
&\quad + \frac{2 \alpha ( \Vert w_0 - w_* \Vert_{I_{0:k^*}}^2 + T \gamma  \Vert w_0 - w_* \Vert_{H_{k^*: \infty}}^2 )}{T \gamma (1 - \gamma \alpha {\rm tr}(H))} \left( \frac{k^*}{T} + T \gamma^2 \sum_{i > k^*} \lambda_i^2 \right).
\end{align*}
$$


详细的推导详见原论文，原论文并且给出了与上述的上界几乎匹配的下界，说明文章给出的泛化风险的界是较为紧的。





