---
title: '特征值不等式和最佳低秩逼近'
toc: true
excerpt_separator: <!--more-->
tags:
  - 高等线性代数
---


从Hermite矩阵的特征值不等式出发，利用奇异值推广到一般矩阵，得到最佳低秩逼近定理。

<!--more-->

同样先假定矩阵均为Hermite矩阵，分析其特征值，后面利用奇异值会推广到一般矩阵。

也即假定 $A, B \in {C}^{n \times n}$  都是 Hermite 矩阵, 并且它们的特征值按照升序排列, 即$\lambda_{1}(A) \leq \lambda_{2}(A) \leq \cdots \leq \lambda_{n}(A), \quad \lambda_{1}(B) \leq \lambda_{2}(B) \leq \cdots \leq \lambda_{n}(B)$



## Weyl不等式

$\lambda_k(A) + \lambda_1(B) \le \lambda_k(A+B) \le \lambda_k(A)+\lambda_n(B)$

或 $\max_k \vert \lambda_k(A) - \lambda_k(B) \vert \le \Vert A-B \Vert_2$

根据Rayleigh商定理证明，可以详见[特征值的变分性质和谱聚类](https://truenobility303.github.io/Spectral-Clustering/).

证明的关键在于子空间的构建，构建$U_A$为$A$的前k个特征向量张成的子空间，$U_B$为$B$的n个特征向量所张成的子空间，$U_{A+B}$为$A+B$的后n+1-k个特征向量张成的子空间，因为$\text{dim}(U_A)+\text{dim}(U_B) +\text{dim}(U_{A+B}) = 2n+1$ ,根据空间维数的关系，三个子空间之间一定存在交集，其中有单位向量$x$, $\lambda_k(A+B) \le x^{\star} (A+B) x \le x^{\star} A x+x^{\star} B x \le \lambda_k(A) + \lambda_n(B)$ 

上面得到Weyl不等式的一边，其另一边同理可得。

实际上，Weyl不等式的一般形式为

$\lambda_{n-j+1}(A) + \lambda_j(B) \le \lambda_i(A+B) \le \lambda_{i+j}(A) + \lambda_{n-j}(B)$， 其证明过程完全类似。

改写得到，$\max_k \vert \lambda_k(A) - \lambda_k(B) \vert \le \Vert A-B \Vert_2$

## Hoffman–Wielandt不等式

参考[大佬博客](https://djalil.chafai.net/blog/2011/12/03/the-hoffman-wielandt-inequality/)

如果说Weyl不等式给出了矩阵特征值与二范数的关系，那么Hoffman–Wielandt不等式给出了矩阵特征值与F范数的关系。

$\sum_k \vert \lambda_k(A) - \lambda_k(B)^2 \vert \le \Vert A-B \Vert_F^2$

证明需要用到矩阵$A,B$的谱分解,

$\Vert A-B\Vert_F^2 = \Vert U^TAU-V^T B V \Vert_F=\Vert V U^TA- B V U^T \Vert_F^2 = \sum_{ij} w_{ij}^2(\lambda_i(A) -\lambda_j(B))^2=\sum_{ij}P_{ij}(\lambda_i(A) -\lambda_j(B))^2, \\ \text{Let }W=VU^T, P_{ij} = W_{ij}^2$

且可知$W$是一个双随机矩阵，且由著名的Birkhoff 定理，双随机矩阵是有限个排列阵的凸组合，则下式在一个凸顶点处取得最小值

$ \min \Vert A-B \Vert_F^2 \ge \min \sum_{ij}Q_{ij}(\lambda_i(A)-\lambda_j(B))^2$

由于$Q$为排列阵，且根据熟知的排序不等式

$ \min \sum_{ij}Q_{ij}(\lambda_i(A)-\lambda_j(B))^2 = \sum_{i}(\lambda_i(A)-\lambda_j(B))^2$

也即得到Hoffman-Wielandt不等式的形式。



## 从特征值到奇异值

根据特征值和奇异值的关系，可以将上述两个不等式应用到一般的矩阵$A$上，只需要定义

$$
\widetilde{A} = \left\( \begin{matrix} 
0 & A \\\
A^{\star} & 0 \\\
\end{matrix} \right\)
$$

则上述矩阵为Hermite矩阵，其其特征值由$A$的特征值或特征值的相反数组成，应用上述不等式得到，

$\sum_k \vert \sigma_k(A) - \sigma_k(B) \vert^2 \le \Vert A-B \Vert_F^2$

$\max_k \vert \sigma_k(A)-\sigma_k(B) \vert^2 \le \Vert A-B \Vert_2^2$



## 最佳低秩逼近

最佳低秩逼近是上述两个不等式最重要的应用，只需找到取等条件即可，下面直接给出结论。

最佳低秩逼近在图像压缩领域等具有重要的作用，其含义是对于一个矩阵仅有较大的奇异值包含了其主要信息。

当限制$\text{rank }(B) \le k$的时候，

$\min \Vert A-B \Vert_F^2 = \sum_{i=1}^k \sigma_i^2(A)$

$\min \Vert A-B \Vert_2^2 = \sigma_k^2(A)$



## Kehan公式

另一个重要的应用是Kehan公式的证明，参考[潘神的博客](https://www.bilibili.com/read/cv8309738?spm_id_from=333.999.0.0)

Kehan公式衡量的是一个矩阵$A$距离一个奇异矩阵的距离，

$\{\Vert E \Vert_2:\text{det }(A+E)=0\}=\sigma_1(A)$

根据最佳低秩逼近理论，矩阵到奇异矩阵的最小距离就是其最小的奇异值，得证。
