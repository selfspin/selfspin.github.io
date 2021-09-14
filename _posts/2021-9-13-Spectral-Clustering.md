---
title: '特征值的变分性质和谱聚类'
toc: true
excerpt_separator: <!--more-->
tags:
  - 高等线性代数

---





从特征值的变分性质出发，推导樊氏迹极小化原理，从而得到谱聚类。

<!--more-->



本小节假定 $A, B \in {C}^{n \times n}$  都是 Hermite 矩阵, 并且它们的特征值按照升序排列, 即

### Rayleigh商定理

$\lambda_{1}(A) \leq \lambda_{2}(A) \leq \cdots \leq \lambda_{n}(A), \quad \lambda_{1}(B) \leq \lambda_{2}(B) \leq \cdots \leq \lambda_{n}(B)$

可以得到下面的Rayleigh商公式，

$
\lambda_{1}(A)=\min_{x \in {C}^{n} } \frac{x^{\star} A x}{x^{\star} x} \\
\lambda_{n}(A)=\max_{x \in {C}^{n} } \frac{x^{\star} A x}{x^{\star} x}
$

定理的证明是显然的，将$A$进行谱分解，$A=\sum_i\lambda_i q_i q_i^T,x=\sum_i x_iq_i$, 则$\frac{x^{\star} A x}{x^{\star} x}=\frac{\sum_i\lambda_ix_i^2}{\sum_i x_i^2} = \sum_i \lambda_i x_i^2 \vert\sum_i x_i^2=1$

上述是一个关于$\lambda_i$的凸组合，其最大最小值是显然的。

Rayleigh商公式对一般的$\lambda_i(A)$也是成立的，和所限定的子空间相关。



### Courant-Fischer 极小极大定理

$\begin{align}\lambda_{k}(A) &=\min_{\mathcal{V} \subset {C}^{n},\text{dim}(\mathcal{V})=k} \max_{x \in \mathcal{V} } \frac{x^{\star} A x}{x^{\star} x} \\
&=\max_{\mathcal{V} \subset {C}^{n} ,\text{dim}(\mathcal{V})=n+1-k } \min_{x \in \mathcal{V} } \frac{x^{\star} A x}{x^{\star} x}\end{align}$

基于上述观察，可以直接得到如上定理，证明略，从子空间的角度出发即可。



### Cauchy 交错定理

若  $C$  是  $A$  的  n-1  阶主子阵,  $C$  的特征值从小到大依次为  $\mu_{1} ,  \mu_{2}, \ldots, \mu_{n-1}$ , 那么

$\lambda_{1} \leq \mu_{1} \leq \lambda_{2} \leq \mu_{2} \leq \cdots \leq \lambda_{n-1} \leq \mu_{n-1} \leq \lambda_{n}$

或者写成: $\lambda_{i}(A) \leq \lambda_{i}(C) \leq \lambda_{i+1}(A)(1 \leq i \leq n-1) $



证明：使用Rayleigh商表示特征值，$C$的特征向量张成的子空间为$U_C$, 由于$C$是$A$的主子阵，将$U_C$中的向量的最后一维置0，并且进行基扩充，可以张成$A$的子空间$U_A$.但这种方法张成的子空间是有限制的，前n-1个特征向量的最后一维为0.该限制导致了定理中的不等号。
$$
\lambda_k(A) = \min_{\mathcal{V} \subset \mathbb{C}^{n} \atop \operatorname{dim}(\mathcal{V})=k} \max _{x \in \mathcal{V} \backslash\{0\}} \frac{x^{\star} A x}{x^{\star} x} \le \min_{\mathcal{V} =U_A   \atop \operatorname{dim}(\mathcal{V})=k} \max _{x \in \mathcal{V} \backslash\{0\}} \frac{x^{\star} A x}{x^{\star} x} = \min_{\mathcal{V} =U_C   \atop \operatorname{dim}(\mathcal{V})=k} \max_{x \in \mathcal{V} \backslash\{0\}} \frac{x^{\star} A x}{x^{\star} x} =\lambda_k(C)
$$
得到定理的一边，$\lambda_i(A) \le \lambda_i(C)$ 

另一边，只需对$-A$和$-C$应用上述定理即可，由于$C$的大小比$A$小1，因此$\lambda_i(C) \le \lambda_{i+1}(A)$ 



- 推论: 若  $\lambda$  是  A  的  m  重特征值, 那么  $\lambda$  至少是  $C$  的m-1重特征值. 该推论直接可得。
- 推广形式 1 : 若  C  是  A  的  n-k  阶主子阵, 那么$\lambda_{i}(A) \leq \lambda_{i}(C) \leq \lambda_{i+k}(A), \quad(1 \leq i \leq n-k)$ .证明同理。

- 推广形式 2 : 若  $X \in {C}^{n \times k}$  是某个  n  阶西矩阵的子矩阵, 即  $X^{\star} X=I_{k}$ , 那 么$\lambda_{i}(A) \leq \lambda_{i}\left(X^{\star} A X\right) \leq \lambda_{i+k}(A), \quad(1 \leq i \leq n-k)$。代入可得，且对$A$进行酉相似变换不改变$A$的特征值。
- 上述几种形式通常都称为 Cauchy 交错定理, 推广形式 2 也称为 Poincaré 隔离定理.



### 樊氏迹极小化原理

$\lambda_{1}(A)+\lambda_{2}(A)+\ldots+\lambda_{k}(A)=\min_{X \in {C} n \times k \atop X^{\star} X=I_{k}} {tr}\left(X^{\star} A X\right)$

证明：应用Cauthy交错定理,$\lambda_i(A) \le \lambda_i(C)$，可得，$\lambda_{1}(A)+\lambda_{2}(A)+\ldots+\lambda_{k}(A) \le \min_{X \in {C} n \times k \atop X^{\star} X=I_{k}} {tr}\left(X^{\star} A X\right)$

$X$取为$A$前$k$个正交特征向量，得到取等条件，因此，$\lambda_{1}(A)+\lambda_{2}(A)+\ldots+\lambda_{k}(A)=\min_{X \in {C} n \times k \atop X^{\star} X=I_{k}} {tr}\left(X^{\star} A X\right)$



## 谱聚类

基于上述准备，可以得到谱聚类。以下内容主要参考自，[社交媒体挖掘](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.4456&rep=rep1&type=pdf) 一书中社区发现一节。



首先，考虑图上的社区发现，将该任务定义为寻找图上的割，使得分割出的社区尽可能均衡。

改进的最小割方法定义了一个目标方程, 并在找割的过程中最小化 (或 最大化）目标函数, 以便找到一个更均衡和自然的数据分割。考虑一个图  G(V, E)  。  G  的一次分割 可记为一个  k  元组  $P=\left(P_{1}, P_{2}, P_{3}, \cdots, P_{k}\right)$ , 其中  $P i \subseteq V, \quad P_{i} \cap P_{j}=\varnothing, U_{i=1}^{k} \dot{P}_{i}=V_{\circ}$ 这样, 比例割和 归一化割的目标函数定义如下：

$
\text { 比例割 }(P)=\frac{1}{k} \sum_{i=1}^{k} \frac{{cut}\left(P_{i}, \bar{P}_{i}\right)}{\vert P_{i}\vert} \\
\text { 归一化割 }(P)=\frac{1}{k} \sum_{i=1}^{k} \frac{{cut}\left(P_{i}, \bar{P}_{i}\right)}{{vol}\left(P_{i}\right)}
$

其中，  $\bar{P}_{i}=V \- P_{i}$  是割集的补集，  $cut\left(P_{i}, \bar{P}\_{i}\right)$  是割的大小, 割集的容量为  ${vol}\left(P_{i}\right)=\sum_{v \in P_{1}} d_{v}$ 。  这两 个目标函数通过除以割集中结点的数量或者是容量 ( 即度的总和) 进行归一化, 使得获得的社区 更加均衡。



比例割和归一化割都可以用矩阵的形式进行公式化表示。假设矩阵  $\boldsymbol{X} \in\{0,1\}^{\vert\eta\vert \times k}$  代表社区关 系矩阵，其中如果结点在社区  j  中, 则  $\boldsymbol{X}_{i, j}=1$; 否则,  $\boldsymbol{X}_{i, j}=0$  。假设  $\boldsymbol{D}={diag}\left(d_{1}, d_{2}, \cdots, d_{n}\right) $ 代表 对角度矩阵。那么矩阵  $\boldsymbol{X}^{\mathrm{T}} \boldsymbol{A X}$  对角线上的第  i  个元素代表社区i内部的边的数量。类似地, 矩阵  $\boldsymbol{X}^{\mathrm{T}} A \boldsymbol{X} $ 对角线上的第  i  个元素代表了与社区i的成员相连的边的数量。因此, 矩阵  $\boldsymbol{X}^{\top}(\boldsymbol{D}-\boldsymbol{A}) \boldsymbol{X}$  对 角线上的第  i  个元素代表了将社区  i  从其他结点分割开的割的边的数目。事实上,  $\boldsymbol{X}^{\mathrm{T}}(\boldsymbol{D}-\boldsymbol{A}) \boldsymbol{X} $ 对 角线上的第  i  个元素即为比例割和归一化割中的  ${cut}\left(P_{i}, \bar{P}\_{i}\right) $ 值。基于此, 对于比例割, 我们有



$\begin{align}
\text {比例割}(P) &=\frac{1}{k} \sum_{i=1}^{k} \frac{{cut}\left(P_{i,} \bar{P}_{i}\right)}{\vert P_{i}\vert} \\
&=\frac{1}{k} \sum_{i=1}^{k} \frac{\boldsymbol{X}_{i}^{\mathrm{T}}(\boldsymbol{D}-\boldsymbol{A}) \boldsymbol{X}_{i}}{\boldsymbol{X}_{i}^{\mathrm{T}} \boldsymbol{X}_{i}} \\
&=\frac{1}{k} \sum_{i=1}^{k} \hat{\boldsymbol{X}}_{i}^{\mathrm{T}}(\boldsymbol{D}-\boldsymbol{A}) \hat{\boldsymbol{X}}_{i}
\end{align}$



其中,  $\hat{\boldsymbol{X}}_{i}=\boldsymbol{X}_{i} /\left(\boldsymbol{X}_{i}^{\mathrm{T}} \boldsymbol{X}_{i}\right)^{1 / 2} $ 。可以采用相似的方法对归一化割进行公式化表示，并获得一个不同 的  $\hat{X}\_{i}$  。为了在比例割和归一化割中用同样的公式化表示求和, 我们可以使用矩阵迹 $(  \left.{tr}(\hat{\boldsymbol{X}})=\sum_{i=1}^{n} \hat{\boldsymbol{X}}\_{i i}\right) $ 。基于矩阵迹, 比例割和归一化割的目标函数可以表示为最小迹问题：$\min_{\hat{x}} Tr\left(\hat{\boldsymbol{X}}^{\mathrm{T}} L \hat{\boldsymbol{X}}\right)$ 其中，  L  是（归一化的）图的拉普拉斯算子（graph Laplacian ）：

$
\boldsymbol{D}-\boldsymbol{A}  \text { (比例割) } \\
\boldsymbol{I}-\boldsymbol{D}^{-1 / 2} \boldsymbol{A D}^{-1 / 2} \text { (归一化割) }
$

可以看出，无论是比例割还是归一化割, 它们的最小化问题都是NP难问题; 因此, 我们需 要使用一些具有松弛条件的近似算法。谱聚类就是这样一种松弛算法:

$
\min_{\dot{X}} Tr\left(\hat{\boldsymbol{X}}^{\mathrm{T}} L \hat{\boldsymbol{X}}\right),\hat{\boldsymbol{X}}^{\mathrm{T}} \hat{\boldsymbol{X}}=I_{k}
$



经过上述松弛转化为樊氏迹极小化问题，取前k个特征向量和特征值即可。



## 应用

谱聚类应用广泛，上述基于社区发现推出谱聚类的算法。

实际上，谱聚类不仅仅可以用在图上，对于一般数据的聚类问题，可以根据数据之间的距离生成权重矩阵（如根据高斯函数，赋予距离较大的数据点更大的权重），构建出图后使用谱聚类。此时谱聚类可以作为一种通用的代替K-Means等方法的聚类手段。
