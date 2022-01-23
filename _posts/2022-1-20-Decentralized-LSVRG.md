---
title: 'Decentralized L-SVRG'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
  - 分布式优化
  - 随机优化
---



论文阅读笔记：[PMGT-VR: A decentralized proximal-gradient algorithmic framework with variance reduction](https://arxiv.org/abs/2012.15010)



<!--more-->

文章可以看作 [L-SVRG](https://truenobility303.github.io/L-SVRG-and-L-Katyusha/) 的去中心化分布式版本，但本文推导的问题不带正则项 $r(x)$， 

因此算法不需要使用 $\textbf{prox}$ 算子，且推导过程借鉴了该文章，但稍有不同，



## Method Overview

关于以下问题，假设共有 $m$ 台去中心化的分布式机器，每台机器最小化 $n$ 个样本的损失函数，


$$
\min f(x) :=  \frac{1}{m} \sum_{i=1}^m f_i(x) = \frac{1}{mn} \sum_{i=1}^m \sum_{j=1}^n f_{ij}(x)
$$



假设每一个损失函数都满足 $L$ -光滑和 $\mu$-强凸性质，也即，



$$
\begin{align}
f_{ij}(x) &\le f_{ij}(y) + \nabla f_{ij}(y)^\top(x-y) + \frac{L}{2} \Vert x- y \Vert^2 \\
f_{ij}(x) &\ge f_{ij}(y) + \nabla f_{ij}(y)^\top(x-y) + \frac{\mu}{2} \Vert x- y \Vert^2
\end{align}
$$



与 [Decentralized AGD](https://truenobility303.github.io/Decentralized-AGD/) 的框架类似，尝试使用FastMix算子使得去中心化算法可以表现出类似中心化算法的行为，取 $p = \frac{1}{n}$ 


$$
\begin{align}

\mathbf{v}_k &= \nabla f_{j}(\mathbf{x}_k) - \nabla f_{j}( \mathbf{w}_k) + \nabla f(\mathbf{w}_k) ,j \sim \mathcal{U}(0,n)\\
\mathbf{w}_{k+1} &= \mathbf{x}_k \text{ With Prob. } p \\
&= \mathbf{w}_k \text{With Prob. } 1-p \\
\mathbf{s}_{k} &= \text{FastMix}(\mathbf{s}_{k-1} + \mathbf{v}_k - \mathbf{v}_{k-1}) \\
\mathbf{x}_{k+1} &= \text{FastMix}(\mathbf{x}_k - \eta \mathbf{s}_k)
\end{align}
$$



定义相应的Lyapunov函数，对于 $\mathcal{D_k}$ 变为关于 $i,j$ 也即所有机器上的所有损失函数的求和，



$$
\begin{align}
\mathcal{V_k} &= \Vert  \bar x_k - \ x_{\ast} \Vert^2 + \frac{4n \eta^2}{m} \mathcal{D_k} , \text{With } \mathcal{D_k} = \mathbb{E}[\Vert \nabla f_j(\mathbf{w}_k) - \mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2]
\end{align}
$$



## Bound for Consensus Error



由于算法希望去中心化的行为与中心化分布式优化的行为相似，因此需要推导 Consensus \mathbb{E}rror的界，对于 $\mathbf{x}_{k+1}$


$$
\begin{align} 
\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert &\le  \rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k - \eta \mathbf{s}_k + \eta \mathbf{1} \cdot \bar s_k \Vert \\
&\le \rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + \rho \eta \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert 
\end{align}
$$


对于 $\mathbf{s}_k$ , 类似地可以有，


$$
\begin{align}
\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert &\le \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + \rho \Vert \mathbf{v}_{k+1} - \mathbf{v}_k  -\mathbf{1} \cdot \bar v_{k+1} + \mathbf{1} \cdot \bar v_k \Vert \\
&\le  \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + \rho \Vert \mathbf{v}_{k+1} - \mathbf{v}_k  \Vert \\
\end{align}
$$


与 [Decentralized AGD](https://truenobility303.github.io/Decentralized-AGD/) 有所不同的是，在SVRG的Variance Reduction框架下，可以证明此处的 $\mathbf{v}_{k+1}$ 与 $\mathbf{v}_k$ 差距不会太大，证明思路采用SVRG的处理类似的手段，常数项会稍微优于原论文的结果


$$
\begin{align}
\mathbb{E} [\Vert \mathbf{v}_{k} \Vert^2] &\le \mathbb{E}[\Vert \nabla f_{j}(\mathbf{x}_k) - \nabla f_{j}( \mathbf{w}_k) + \nabla f(\mathbf{w}_k) \Vert^2] \\
&\le 2\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) -\mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2 ] +2 \mathbb{E}[\Vert \mathbf{1} \cdot\nabla f_j(x_{\ast})   - \nabla f_{j}( \mathbf{w}_k) + \nabla f(\mathbf{w}_k) \Vert^2] \\
&\le 2\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) -\mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2 ] +2 \mathbb{E}[\Vert \nabla f_{j}( \mathbf{w}_k) - \mathbf{1} \cdot\nabla f_j(x_{\ast})   \Vert^2] \\
&= 2\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) -\mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2 ] +2 \mathcal{D_k} \\
&\le4\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) - \mathbf{1} \cdot\nabla f_j(\bar x_k)\Vert^2 ] +4 \mathbb{E}[\Vert \mathbf{1} \cdot \nabla f_j(\bar x_k)-  \mathbf{1} \cdot \nabla f_j(x_{\ast}) \Vert^2 ] +2 \mathcal{D_k} \\
&\le 4L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 4L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_k} 
\end{align}
$$


采用原论文的方法，取平方并且对随机采样取期望可以得到，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert^2] 
&\le 2\rho^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2\rho^2 \eta^2 \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2  \\
\end{align}
$$


以及，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert^2] 
&\le  2\rho^2 \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert^2 + 2\rho^2 \mathbb{E}[\Vert \mathbf{v}_{k+1} - \mathbf{v}_k  \Vert^2] \\
&\le 2 \rho^2 \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert^2  + 4 \rho^2 \mathbb{E} [\Vert \mathbf{v}_{k+1} \Vert^2] + 4 \rho^2 \mathbb{E}[\Vert \mathbf{v}_k \Vert^2] \\
&\le 2 \rho^2 \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert^2 + 4 \rho^2(4L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 4L^2 \Vert \mathbf{1} \cdot\bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_k}) \\
&\quad +4 \rho^2(4L^2 \Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert^2 + 4L^2 \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_{k+1}}) \\
&\le 2 \rho^2 \Vert  \mathbf{s}_{k} - \mathbf{1} \cdot \bar s_k \Vert^2 + 4 \rho^2(4L^2\Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 4L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_k}) \\
&\quad +4 \rho^2( (8\rho^2L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 8\rho^2 L^2\eta^2 \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 ) + 4L^2 \Vert \mathbf{1} \cdot\bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_{k+1}}) \\
&= (2\rho^2+32\rho^4L^2 \eta^2) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 + (16\rho^2L^2+ 32\rho^4 L^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2\\
&\quad + 8 \rho^2 \mathcal{D_k} + 8 \rho^2\mathcal{D_{k+1}} + 16\rho^2 L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 16\rho^2 L^2 \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert^2
\end{align}
$$



原论文通过这种方式建立起了相应的关于Consensus Error的不等式线性系统，但是常数项较为复杂，下面给出另外一种方式，



对于不平方的范数，


$$
\begin{align} 
\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert 
&\le \rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + \rho \eta \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert 
\end{align}
$$


利用三角不等式，
$$
\begin{align}
\mathbb{E} [\Vert \mathbf{v}_k \Vert] &\le \sqrt{\mathbb{E} [\Vert \mathbf{v}_k \Vert^2 ]}  \\
&\le \sqrt{4L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 4L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 2\mathcal{D_k} } \\
&\le 2L \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + 2L \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert + \sqrt {2 \mathcal{D_k}} \\
\end{align}
$$


类似地，


$$
\begin{align}
\mathbb{E} [\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert] 
&\le  \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + \rho \mathbb{E}[\Vert \mathbf{v}_{k+1} - \mathbf{v}_k  \Vert] \\
&\le \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + \rho \mathbb{E} [\Vert \mathbf{v}_k \Vert] + \rho \mathbb{E} [\Vert \mathbf{v}_{k+1}\Vert] \\
&\le \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + 2L\rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + 2L\rho \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert + \rho \sqrt {2 \mathcal{D_k}} \\ 
&\quad + 2L\rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_{k+1} \Vert + 2L\rho \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert + \rho\sqrt {2 \mathcal{D_{k+1}}} \\
&\le \rho \Vert  \mathbf{s}_{k} - \mathbf{1} \bar s_k \Vert + 2L\rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + 2L\rho \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert + \rho \sqrt {2 \mathcal{D_k}} \\ 
&\quad + 2L\rho  ^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + 2L\rho^2 \eta \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert  + 2L\rho \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert + \rho\sqrt {2 \mathcal{D_{k+1}}} \\
&\le (2L\rho + 2L \rho^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +(\rho+2L \rho^2 \eta) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \\
&\quad + 2L \rho \Vert \mathbf{1} \cdot\bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert + 2 L\rho \Vert \mathbf{1} \cdot\bar x_{k} - \mathbf{1} \cdot x_{\ast} \Vert + \rho\sqrt {2 \mathcal{D_{k+1}}}+ \rho\sqrt {2 \mathcal{D_{k}}}
\end{align}
$$


## Bound for Lyapunov Function



利用上述推导的不等式线性系统，尝试推导关于Lyapunov 函数的递推不等式，




$$
\begin{align}
\mathbb{E}[\Vert  \bar x_{k+1} -  x_{\ast} \Vert^2 &=\mathbb{E}[\Vert  \bar x_{k} - \eta \bar s_k -  x_{\ast} \Vert^2] \\
&= \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta \mathbb{E}[ \bar s_k^\top ( \bar x_k -  x_{\ast})] + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\
& = \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta \mathbb{E}[ \bar v_k^\top ( \bar x_k -  x_{\ast})] + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\
&= \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta \bar g_k^\top ( \bar x_k -  x_{\ast}) + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] , \text{Let } \bar g_k = \frac{1}{m} \sum_{i=1}^m \nabla f_i(\mathbf{x}_k)\\
\end{align}
$$




继续 Bound 上式中的项，


$$
\begin{align}
\Vert \bar g_k - \nabla f(\bar x_k) \Vert &= \Vert \frac{1}{m} \sum_{i=1}^m \nabla f_i(\mathbf{x}_k^{(i)}) - \nabla f_i( \bar x_k) \Vert \\
&\le  \frac{1}{\sqrt m } \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert
\end{align}
$$


以及，


$$
\begin{align}
\mathbb{E} [ \Vert \bar s_k \Vert^2] &= \mathbb{E}[ \Vert \bar v_k \Vert^2] \\ 
&\le \frac{1}{m} \mathbb{E} [\Vert \mathbf{v_k} \Vert^2] \\ 
&\le \frac{1}{m} (4\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) - \mathbf{1} \cdot\nabla f_j(\bar x_k)\Vert^2 ] +4 \mathbb{E}[\Vert \mathbf{1} \cdot \nabla f_j(\bar x_k)-  \mathbf{1} \cdot \nabla f_j(x_{\ast}) \Vert^2 ] +2 \mathcal{D_k})  \\
&\le \frac{4L^2}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 8 L \mathbb{E} [f_j(\bar x_k) - f_j(x_{\ast})] +  \frac{2}{m} \mathcal{D_k} \\ 
&= \frac{4L^2}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 8 L (f(\bar x_k) - f(x_{\ast})) +  \frac{2}{m} \mathcal{D_k} \\
\end{align}
$$


因而利用到 $\bar g_k $ 与 $\nabla f(\bar x_k)$ 差距不大，


$$
\begin{align}
\mathbb{E}[\Vert  \bar x_{k+1} -  x_{\ast} \Vert^2 
&\le \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta \bar g_k^\top ( \bar x_k -  x_{\ast}) + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\
&\le \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta \nabla f(\bar x_k)^\top (\bar x_k - x_{\ast}) + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\
&\le  \Vert  \bar x_k -  x_{\ast} \Vert^2 -2 \eta (f(\bar x_k) -f(x_{\ast})  + \frac{\mu}{2} \Vert x_{\ast}- \bar x_k \Vert^2) + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\
&\le (1- \eta \mu) \Vert  \bar x_k -  x_{\ast} \Vert^2 -2  \eta (f(\bar x_k) - f(x_{\ast})) + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\ 
\end{align}
$$


而对于 $\mathcal{D_k}$ 的递推不等式，利用期望的性质，


$$
\begin{align}
\mathbb{E}[\mathcal{D_{k+1}}] &= (1-p) \mathcal{D_k} + p\mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) - \mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2] \\
&\le (1-p ) \mathcal{D_k} + 2p \mathbb{E}[\Vert \nabla f_j(\mathbf{x}_k) - \mathbf{1} \cdot\nabla f_j(\bar x_{k}) \Vert^2] + 2p \mathbb{E}[\Vert \mathbf{1} \cdot \nabla f_j(\bar x_k) - \mathbf{1} \cdot\nabla f_j(x_{\ast}) \Vert^2]\\
&\le (1-p) \mathcal{D_k} + 2p L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2p m \mathbb{E}[\Vert  \nabla f_j(\bar x_k) - \nabla f_j(x_{\ast}) \Vert^2]\\
&\le (1-p) \mathcal{D_k} + 2p L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2p m L(f(\bar x_k) - f(x_{\ast}))\\
\end{align}
$$




回顾所定义的Lyapunov 函数,


$$
\mathcal{V_k} = \Vert  \bar x_k - \ x_{\ast} \Vert^2 + \frac{4n \eta^2}{m} \mathcal{D_k}
$$


代入之前的引理可以得到，


$$
\begin{align}
\mathbb{E}[\mathcal{V_{k+1}}] &= \mathbb{E}[ \Vert \bar x_{k+1} - x_{\ast} \Vert^2 + \frac{4 n \eta^2}{m} \mathcal{D_{k+1}}] \\
&\le (1- \eta \mu) \Vert  \bar x_k -  x_{\ast} \Vert^2 -2  \eta (f(\bar x_k) - f(x_{\ast})) + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \eta^2 \mathbb{E}[\Vert   \bar s_k \Vert^2] \\  
&\quad + \frac{4n \eta^2}{m}((1-p) \mathcal{D_k} + 2p L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2p m L(f(\bar x_k) - f(x_{\ast})))\\ 
&\le (1- \eta \mu) \Vert  \bar x_k -  x_{\ast} \Vert^2 -2  \eta (f(\bar x_k) - f(x_{\ast})) + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert  \\  
&\quad + \frac{4n \eta^2}{m}((1-p) \mathcal{D_k} + 2p L^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2p m L(f(\bar x_k) - f(x_{\ast})))\\ 
&\quad + \eta^2(\frac{4L^2}{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 8 L (f(\bar x_k) - f(x_{\ast})) +  \frac{2}{m} \mathcal{D_k})\\
&= (1-n\mu) \Vert \bar x_k - x_{\ast} \Vert^2 +(16\eta^2 L - 2 \eta)(f(\bar x_k) - f(x_{\ast})) + (1- \frac{p}{2}) \frac{4 n \eta^2}{m}  \mathcal{D_k} \\
&\quad + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \frac{12\eta^2 }{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 
\end{align}
$$



设置足够小的步长 $\eta = \frac{1}{8L}$ 则可以得到，

$$
\begin{align}
\mathbb{E}[\mathcal{V_{k+1}}] &\le  (1-\frac{\mu}{8L}) \Vert \bar x_k - x_{\ast} \Vert^2 +(1- \frac{p}{2})  \frac{4 n \eta^2}{m} \mathcal{D_k} + 2 \eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \frac{12\eta^2 }{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 \\
&\le \max(1-\frac{\mu}{8L}, 1- \frac{1}{2n}) \mathcal{V_k} + 2\eta \Vert \bar g_k - \nabla f(\bar x_k) \Vert \Vert \bar x_k - x_{\ast} \Vert + \frac{12\eta^2 }{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 \\
&\le \max(1-\frac{\mu}{8L}, 1- \frac{1}{2n}) \mathcal{V_k} + \frac{2\eta}{\sqrt m}\Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert \sqrt{\mathcal{V_k}} +    \frac{12\eta^2 }{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2
\end{align}
$$



从而用Consensus Error控制住了Lyapunov函数



## Proof for Convergence



将关于Consensus Error的线性不等式写为矩阵的形式，首先利用Lyapunov函数进行化简，


$$
\begin{align}
2L  \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert + \sqrt {2 \mathcal{D_{k}}} &\le \sqrt{8L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 4 \mathcal{D_k} } \\
&\le 8L \sqrt m \sqrt { \frac{1}{8} \Vert \bar x_k - x_{\ast} \Vert^2+ \frac{1}{16L^2m} \mathcal{D_k}} \\
&\le 8L \sqrt m \sqrt{\Vert \bar x_k - x_{\ast} \Vert^2+ \frac{n}{16L^2m} \mathcal{D_k}} \\
&=8L\sqrt m \sqrt{\Vert  \bar x_k - \ x_{\ast} \Vert^2 + \frac{4n \eta^2}{m} \mathcal{D_k}} \\
&= 8L \sqrt m \sqrt{\mathcal{V_k}}
\end{align}
$$




因而,


$$
\begin{align}
\mathbb{E} [\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert] 
&\le (2L\rho + 2L \rho^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +(\rho+2L \rho^2 \eta) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert \\
&\quad + 2L \rho \Vert \mathbf{1} \cdot\bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert + 2 L\rho \Vert \mathbf{1} \cdot\bar x_{k} - \mathbf{1} \cdot x_{\ast} \Vert + \rho\sqrt {2 \mathcal{D_{k+1}}}+ \rho\sqrt {2 \mathcal{D_{k}}} \\
&\le (2L\rho + 2L \rho^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert +(\rho+2L \rho^2 \eta) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert +8L \rho \sqrt m (\sqrt{\mathcal{V_k}} + \sqrt{\mathcal{V_{k+1}}})
\end{align}
$$


以及之前就得到的，


$$
\begin{align} 
\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert 
&\le \rho \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert + \rho \eta \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert 
\end{align}
$$


从而得到矩阵形式，


$$
\begin{align}
\begin{pmatrix}
\mathbb{E} [\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_k \Vert] \\
\mathbb{E} [\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_k \Vert] \\
\end{pmatrix}
&\le 
\begin{pmatrix}
\rho & \rho \eta \\
2L \rho +2L \rho^2 & \rho+ 2L \rho^2 \\
\end{pmatrix}
+ 
8L \rho \sqrt m
\begin{pmatrix}
0 \\
\sqrt{\mathcal{V_k}} + \sqrt{\mathcal{V_{k+1}}}
\end{pmatrix}
\end{align}
$$


使用 $\mathbf{z}_k, A$ 简化记号，


$$
\begin{align}
\mathbb{E}[\mathbf{z}_{k+1}] &\le A \mathbf{z}_k + 8L\rho \sqrt m (0, \sqrt{\mathcal{V_k}} +\sqrt {\mathcal{V_{k+1}}})^\top \\
\mathbb{E}[\mathbf{z}_{k+1}] &\le A^{k+1} \mathbf{z}_k + 8 L\rho \sqrt m\sum_{i=0}^k A^{k-i} (\sqrt{\mathcal{V_i}} +\sqrt {\mathcal{V_{i+1}}}) 
\end{align}
$$


想要使得矩阵 $A$ 的谱半径小于 $\frac{1}{2}$, 只需要令FastMix算子产生的结果精度达到给定要求，


$$
\begin{align}
\Vert A \Vert_2 &\le \Vert A \Vert_F \le \sum_{ij} A_{ij} \le  \rho D_3 \le \frac{1}{2} \\
\rho &\le \frac{1}{2} D_3 = 1 +2L + \frac{1}{16L},\text{With } D_3 = 2 + 4L + \frac{1}{8L}
\end{align}
$$




回顾之前得到的对于Lyapunov函数的递推上界，


$$
\begin{align}
\mathbb{E}[\mathcal{V_{k+1}}] 
&\le \max(1-\frac{\mu}{8L}, 1- \frac{1}{2n}) \mathcal{V_k} + \frac{2\eta}{\sqrt m}\Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert \sqrt{\mathcal{V_k}} +    \frac{12\eta^2 }{m} \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 \\
&\le (1-\alpha )\mathcal{V_k} + \frac{2\eta}{\sqrt m}\Vert \mathbf{z}_k  \Vert \sqrt{\mathcal{V_k}} +    \frac{12\eta^2 }{m} \Vert \mathbf{z}_k \Vert^2, \text{Let } \alpha = \min(\frac{\mu}{8L}, \frac{1}{2n}) \le \frac{1}{2} \\
\end{align}
$$


希望使用归纳法得到最终的收敛性证明，观察递推边界，


$$
\begin{align}
\mathbb{E} [ \mathcal{V_1}] &\le (1-\alpha )\mathcal{V_0} + \frac{2\eta}{\sqrt m}\Vert \mathbf{z}_0  \Vert \sqrt{\mathcal{V_0}} +    \frac{12\eta^2 }{m} \Vert \mathbf{z}_0 \Vert^2 \\
&\le (1-\alpha) \mathcal{V_0} + \frac{\alpha}{2} \mathcal{V_0} + \frac{2 \eta^2 }{\alpha m } \Vert \mathbf{z}_0 \Vert^2 + \frac{12 \eta^2}{m} \Vert \mathbf{z}_0 \Vert^2 \\
&=(1-\frac{\alpha}{2}) (\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0 \Vert^2) ,\text{By } \Vert \mathbf{z}_0 \Vert = 0  
\end{align}
$$


因此不妨假设对于 $i \le k$ 的时候都成立，


$$
\begin{align}
\mathbb{E} [ \mathcal{V_k}] &\le (1-\frac{\alpha}{2})^k (\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0  \Vert^2) \\
\mathbb{E}[\sqrt {\mathcal{V_k}}] &\le ( \sqrt{1-\frac{\alpha}{2}})^k  \sqrt{\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0  \Vert^2} \\
&\le ( \sqrt{1-\frac{\alpha}{2}})^k  (\sqrt{\mathcal{V_0}} + \frac{1}{\sqrt m} \Vert \mathbf{z}_0 \Vert) \\
\end{align}
$$


利用矩阵 $A$ 的谱半径，尝试证明递推式成立，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{z}_{k+1} \Vert] &\le \rho D_3 2^{-k} \Vert \mathbf{z}_0 \Vert + 8 L\rho \sqrt m\sum_{i=0}^k 2^{-(k-i)} (\sqrt{\mathcal{V_i}} +\sqrt {\mathcal{V_{i+1}}}) \\
&\le \rho D_3 2^{-k}\Vert \mathbf{z}_0 \Vert + 8L \rho \sqrt m (1+2 \sqrt{1- \frac{\alpha}{2}} )\sum_{i=0}^k 2^{-(k-i)} ( \sqrt{1-\frac{\alpha}{2}})^i (\sqrt{\mathcal{V_0}} + \frac{1}{\sqrt m} \Vert \mathbf{z}_0 \Vert) \\
&\le 2\rho D_3 2^{-{k+1}} \Vert  \mathbf{z}_0 \Vert+ 24 L\rho \sqrt m \frac{2 (\sqrt{1 - \frac{\alpha}{2}})^{k+1} - 2^{-k}}{2\sqrt{1 - \frac{\alpha}{2} }-1}  (\sqrt{ \mathcal{V_0}} + \frac{1}{\sqrt{m}} \Vert \mathbf{z}_0 \Vert) \\
&\le (\sqrt{1- \frac{\alpha}{2}})^{k+1}\rho (2  D_3 \Vert  \mathbf{z}_0 \Vert + 72 L  \sqrt m  (\sqrt{ \mathcal{V_0}} + \frac{1}{\sqrt{m}} \Vert \mathbf{z}_0 \Vert)) \\
&\le(\sqrt{1- \frac{\alpha}{2}})^{k+1} \rho (2D_3+72L) \sqrt m(\sqrt {\mathcal{V_0}}+ \frac{1}{\sqrt m} \Vert \mathbf{z}_0 \Vert) \\
\end{align}
$$


以及根据条件独立性，


$$
\begin{align}
\mathbb{E} [ \sqrt{\mathcal{V_k} } \Vert \mathbf{z}_k \Vert] &\le \sqrt 2 (1-\frac{\alpha}{2})^k \rho(2D_3 + 72L) \sqrt m (\mathcal{V_0} + \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2) \\
&\le  (1-\frac{\alpha}{2})^k \rho(4D_3 + 144L) \sqrt m (\mathcal{V_0} + \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2) \\
\end{align}
$$


我们几乎已经成功了，但此时仍然剩下  $\mathbb{E}[ \Vert \mathbf{z}_k \Vert^2]$ 未被Bound住，但一阶矩的Bound并不能推导二阶矩的Bound，因此下面建立关于二阶矩的不等式线性系统，



也即还是回归到的原论文的方法，



$$
\begin{align}
\mathbb{E}[\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert^2] 
&\le 2\rho^2 \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2 + 2\rho^2 \eta^2 \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2  \\
\end{align}
$$


以及，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert^2] 
&\le  (2\rho^2+32\rho^4L^2 \eta^2) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 + (16\rho^2L^2+ 32\rho^4 L^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2\\
&\quad + 8 \rho^2\mathcal{D_k} + 8 \rho^2 \mathcal{D_{k+1}} + 16\rho^2 L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 16\rho^2 L^2 \Vert \mathbf{1} \cdot \bar x_{k+1} - \mathbf{1} \cdot x_{\ast} \Vert^2 \\
\end{align}
$$



采用处理一阶不等式线性系统的方式进行处理二阶系统，
$$
\begin{align}
16L^2 \Vert \mathbf{1} \cdot \bar x_k - \mathbf{1} \cdot x_{\ast} \Vert^2 + 8 \mathcal{D_k} &= 16m L^2 \Vert \bar x_k - x_{\ast} \Vert^2 + 8 \mathcal{D_k} \\
&\le 16m L^2 \Vert \bar x_k - x_{\ast} \Vert^2 + 8 n\mathcal{D_k} \\
&= 112 mL^2( \frac{1}{8} \Vert \bar x_k - x_{\ast} \Vert^2 + \frac{n}{16mL^2} \mathcal{D_k} ) \\
&\le 112 mL^2(\Vert \bar x_k - x_{\ast} \Vert^2 + \frac{4n \eta^2}{m} \mathcal{D_k}) \\
&= 112 mL^2 \mathcal{V_k}
\end{align}
$$


因而，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert^2] 
&\le  (2\rho^2+32\rho^4L^2 \eta^2) \Vert \mathbf{s}_k - \mathbf{1} \cdot \bar s_k \Vert^2 + (16\rho^2L^2+ 32\rho^4 L^2) \Vert \mathbf{x}_k - \mathbf{1} \cdot \bar x_k \Vert^2+ 112 m\rho^2L^2 \mathcal{V_k} + 112m \rho^2 L^2\mathcal{V_{k+1}}\\
\end{align}
$$




进一步，


$$
\begin{align}
\begin{pmatrix}
\mathbb{E}[\Vert \mathbf{x}_{k+1} - \mathbf{1} \cdot \bar x_{k+1} \Vert^2 ] \\
\mathbb{E}[\Vert \mathbf{s}_{k+1} - \mathbf{1} \cdot \bar s_{k+1} \Vert^2 ]
\end{pmatrix}
\le 
\begin{pmatrix}
2 \rho^2 & 2\rho^2 \eta^2 \\
16 \rho^2 L^2 + 32 \rho^4 L^2 & 2\rho^2+32\rho^4L^2 \eta^2 
\end{pmatrix}
\begin{pmatrix}
\mathbb{E}[\Vert \mathbf{x}_{k} - \mathbf{1} \cdot \bar x_{k} \Vert^2 ] \\
\mathbb{E}[\Vert \mathbf{s}_{k} - \mathbf{1} \cdot \bar s_{k} \Vert^2 ]
\end{pmatrix}
+ 
112 m\rho^2L^2
\begin{pmatrix}
0 \\
\mathcal{V_k} + \mathcal{V_{k+1}}
\end{pmatrix}
\end{align}
$$




令不等式线性系统中的转移矩阵为 $B$, 同样希望其谱半径小于 $\frac{1}{2}$, 只需要，


$$
\begin{align}
\Vert B \Vert_2 &\le \Vert B \Vert_F \le \sum_{ij} B_{ij} \le \rho^2 D_4 \le \frac{1}{2} \\
\rho^2 D_4 &\le \frac{1}{2},\text{Let } D_4 = 4+ \frac{97L^2}{2} + \frac{1}{32L^2}  \\
\rho &\le \sqrt{\frac{1}{2 D_4} } = \sqrt{\frac{1}{8+ 97L^2+ \frac{1}{16L^2} } }  \\
\end{align}
$$


回顾归纳假设，
$$
\mathbb{E} [ \mathcal{V_k}] \le (1-\frac{\alpha}{2})^k (\mathcal{V_0}+ \frac{1}{m} \Vert \mathbf{z}_0  \Vert^2) \\
$$


则根据归纳假设，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{z}_{k+1} \Vert^2] &\le\rho^2 D_4 2^{-k} \Vert \mathbf{z}_0 \Vert^2 + 112 \rho^2 mL^2 \sum_{i=0}^k2^{i-k} \mathcal{V_i}  + \mathcal{V_{i+1}}\\
&\le 2\rho^2 D_4 2^{-{k+1}} \Vert \mathbf{z}_0 \Vert^2 + 112 \rho^2 mL^2(1-\frac{\alpha}{2}) \sum_{i=0}^k 2^{i-k}(1-\frac{\alpha}{2})^{i} (\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0 \Vert^2) \\
&\le 2\rho^2 D_4 2^{-{k+1}} \Vert \mathbf{z}_0 \Vert^2 + 112 \rho^2 mL^2 \sum_{i=0}^k 2^{i-k}(1-\frac{\alpha}{2})^{i} (\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0 \Vert^2) \\
&\le 2\rho^2D_4 (1-\frac{\alpha}{2})^{k+1} \Vert \mathbf{z}_0 \Vert^2 + 112 \rho^2 \frac{2(1-\frac{\alpha}{2})^{k+1}- 2^{-k}} {1-\alpha}(\mathcal{V_0} + \frac{1}{m }\Vert \mathbf{z}_0 \Vert^2) \\
&\le \rho^2 m(1-\frac{\alpha}{2})^{k+1} (2D_4 \Vert \mathbf{z}_0 \Vert^2 + 448L^2(\mathcal{V_0}+ \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2)) \\
&\le \rho^2 m(1-\frac{\alpha}{2})^{k+1}(2D_4+ 448L^2) (\mathcal{V_0}+ \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2)
\end{align}
$$


根据归纳假设总共得到了，


$$
\begin{align}
\mathbb{E}[\Vert \mathbf{z}_{k+1} \Vert] 
&\le(\sqrt{1- \frac{\alpha}{2}})^{k+1} \rho (2D_3+72L) \sqrt m(\sqrt {\mathcal{V_0}}+ \frac{1}{\sqrt m} \Vert \mathbf{z}_0 \Vert) \\
\mathbb{E} [ \sqrt{\mathcal{V_k} } \Vert \mathbf{z}_k \Vert] 
&\le  (1-\frac{\alpha}{2})^k \rho(4D_3 + 144L) \sqrt m (\mathcal{V_0} + \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2) \\
\mathbb{E}[\Vert \mathbf{z}_{k+1} \Vert^2] 
&\le \rho^2 (1-\frac{\alpha}{2})^{k+1}(2D_4+ 448L^2)m (\mathcal{V_0}+ \frac{1}{m} \Vert \mathbf{z}_0 \Vert^2)
\end{align}
$$


取所有的条件期望并且递推，


$$
\begin{align}
\mathbb{E}[\mathcal{V_{k+1}}] 
&\le (1-\alpha )\mathcal{V_k} + \frac{2\eta}{\sqrt m}\Vert \mathbf{z}_k  \Vert \sqrt{\mathcal{V_k}} +    \frac{12\eta^2 }{m} \Vert \mathbf{z}_k \Vert^2 \\
&\le(1-\frac{\alpha}{2}) \mathcal{V_k} +  \frac{2\eta}{\sqrt m}\Vert \mathbf{z}_k  \Vert \sqrt{\mathcal{V_k}} +    \frac{12\eta^2 }{m} \Vert \mathbf{z}_k \Vert^2 \\
&\le (1-\frac{\alpha}{2})^{k+1} (\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0  \Vert^2) + 2 \eta  \rho(1-\frac{\alpha}{2})^k(4 D_3 + 144L)(\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0  \Vert^2) + 12 \eta^2 \rho^2(1-\frac{\alpha}{2})^k(2D_4+448L^2)(\mathcal{V_0}+ \frac{1}{m} \Vert\mathbf{z}_0  \Vert^2)\\
&\le (1-\frac{\alpha}{2})^{k+1}(\mathcal{V_0}+ \frac{1}{m }\Vert \mathbf{z}_0 \Vert^2) \\
\text{Let } 1-\frac{\alpha}{2} &\ge \frac{\alpha}{2} \ge \max(2 \eta \rho(4D_3+144L), 12 \eta^2 \rho^2(2D_4+448L^2))
\end{align}
$$


从而证明了只需要选取合适的精度 $\rho$ , 该去中心化的算法将有与中心化算法类似的表现，

为了达到 $\epsilon$-近似解，忽略 $\log$ 项后其计算复杂度和通信复杂度为，


$$
\begin{align}
T &= \tilde O( (\kappa + n) \log \frac{1}{\epsilon}) \\
C &= \tilde O(  \frac{1}{\sqrt {1 -\lambda_2(W)}} (\kappa + n) \log \frac{1}{\epsilon}) \\
\end{align}
$$


