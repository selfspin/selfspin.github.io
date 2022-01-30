---
title: 'Linear Coupling and AGM'
toc: true
excerpt_separator: <!--more-->
tags:
  - 优化
---



论文阅读笔记：[Linear Coupling: An Ultimate Unifification of Gradient and Mirror Descent](https://arxiv.org/abs/1407.1537)



<!--more-->





## Gradient Descent 



对于 $L$- 光滑的凸函数，使用步长为 $\frac{1}{L}$ 的 [梯度下降算法](https://truenobility303.github.io/CG/)，也即，可


$$
\begin{align}
x_{k+1} &= x_k - \frac{1}{L} \nabla f(x_k)
\end{align}
$$


此时梯度下降算法可以保证下面的关键式子成立，


$$
\begin{align}
f(x_{k+1}) &= f(x_k - \frac{1}{L} \nabla f(x_k)) \\
&\le f(x_k) - \frac{1}{L} \Vert \nabla f(x_k) \Vert^2 + \frac{1}{2L} \Vert \nabla f(x_k) \Vert^2 \\
&= f(x_k) - \frac{1}{2L} \Vert \nabla f(x_k) \Vert^2
\end{align}
$$



对于这个关键的式子可以得到梯度下降算法的收敛率证明，

$$
\begin{align}
\Vert \nabla f(x_k) \Vert^2 &\le 2L(\mathcal{V}_k - \mathcal{V}_{k+1}), \text{Let } \mathcal{V}_k = f(x_k) - f(x_{\ast}) \\
\min_k \Vert \nabla f(x_k) \Vert^2 &\le \frac{2L(\mathcal{V_k}- \mathcal{V}_0)}{k} \le \frac{2L \mathcal{V}_k}{k}  
\end{align}
$$



因此其对于梯度的范数来说，收敛率为 $\mathcal{O}(\frac{1}{k})$, 



对于距离最优值的距离，


$$
\mathcal{V}_k = f(x_k) - f(x_{\ast})
$$


也具有相同的收敛率，且证明基于上述关键引理。



利用到，


$$
\begin{align}
f(x_{k+1}) - f(x_{\ast}) &\le f(x_k) - f(x_{\ast}) - \frac{1}{2L} \Vert \nabla f(x_k) \Vert^2 \\
&\le \nabla f(x_k)^\top(x_{k} - x_{\ast}) - \frac{1}{2L} \Vert \nabla f(x_k) \Vert^2 \\ 
&=\frac{L}{2} (\Vert x_k - x_{\ast} \Vert^2 - \Vert x_k - x_{\ast} - \frac{1}{L} \nabla f(x_k) \Vert^2) \\
&=\frac{L}{2} ( \Vert x_k - x_{\ast} \Vert^2 - \Vert x_{k+1} - x_{\ast} \Vert^2)
\end{align}
$$


因此上述式子告诉我们，关于最优点的距离也在下降，也即，


$$
\begin{align}
f(x_{k+1}) &\le f(x_k) \\
\Vert x_{k+1} - x_{\ast} \Vert &\le \Vert x_k - x_{\ast} \Vert
\end{align}
$$


上述两个单调性性质是梯度下降的重要性质，且可以推出梯度下降算法的收敛率，递推求和得到，


$$
\begin{align}
f(x_k) - f(x_{\ast}) \le \frac{L\Vert x_0 - x_{\ast} \Vert^2}{2k}
\end{align}
$$


因此达到 $\epsilon$- 最优解的复杂度为 $\mathcal{O}(\frac{1}{k})$. 



## Mirror Descent



对于镜像梯度下降算法来说，对于其2范数的特例情况，和普通梯度下降更新公式相同，


$$
x_{k+1} = x_k - \alpha_k \nabla f(x_k)
$$




关键的引理来自于，


$$
\begin{align}
\alpha_k (f(x_k) - f(x_{\ast})) &\le \alpha_k \nabla f(x_k)^\top(x_k - x_{\ast}) \\
&= \alpha_k \nabla f(x_k)^\top(x_k -x_{k+1}) + \alpha_k \nabla f(x_k)^\top(x_{k+1} - x_{\ast}) \\
&\le \alpha_k \nabla f(x_k)^\top(x_k -x_{k+1}) + (x_k - x_{k+1})^\top(x_{k+1} - x_{\ast}) \\
&= \alpha_k \nabla f(x_k)^\top(x_k -x_{k+1}) + \frac{1}{2} \Vert x_{k} - x_{\ast} \Vert^2 - \frac{1}{2} \Vert x_k -x_{k+1} \Vert^2  - \frac{1}{2} \Vert x_{k+1} - x_{\ast} \Vert^2 \\
&= [\alpha_k \nabla f(x_k)^\top(x_k -x_{k+1}) - \frac{1}{2} \Vert x_k -x_{k+1} \Vert^2 ] + \frac{1}{2} \Vert x_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert x_{k+1} - x_{\ast} \Vert^2 \\
&\le \frac{\alpha_k^2}{2} \Vert \nabla f(x_k) \Vert^2  + \frac{1}{2} \Vert x_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert x_{k+1} - x_{\ast} \Vert^2 \\
\end{align}
$$


据此可以得到相同的 $\mathcal{O}(\frac{1}{k})$ 的收敛率结果，



## Linear Coupling



Linear Coupling可以看作Gradient Descent 和Mirror Descent的线性组合，基于的观察是Gradient Descent当梯度大的时候可以获得大的函数值下降，而Mirror Descent当梯度小的时候更接近最优解，而其线性组合将同时具有两者的优点，在2范数的情况下，算法为，


$$
\begin{align}
x_{k} &= \tau_kz_k + (1- \tau_k) y_k \\
y_{k+1} &= x_{k} - \frac{1}{L} \nabla f(x_{k}) \\
z_{k+1} &=x_{k } - \alpha_k \nabla f(x_k)
\end{align}
$$



Mirror Descent中的引理告诉我们，


$$
\begin{align}
\alpha_k  \nabla f(x_k)^\top(z_k - x_{\ast}) &\le \frac{\alpha_k^2}{2} \Vert \nabla f(x_k) \Vert^2   + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 \\
\end{align}
$$


而Gradient Descent 中的引理告诉我们，


$$
\begin{align}
f(x_k ) - f(y_{k+1}) \ge \frac{1}{2L} \Vert \nabla f(x_k ) \Vert^2   
\end{align}
$$


因此联合起来得到了，


$$
\begin{align}
\alpha_k  \nabla f(x_k)^\top(z_k - x_{\ast}) &\le \alpha_k^2L (f(x_k) - f(y_{k+1}))   + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 \\
\end{align}
$$


在此基础上，


$$
\begin{align}
\alpha_{k+1}(f(x_k) - f(x_{\ast})) &\le \alpha_{k+1} \nabla f(x_k)^\top(x_k - x_{\ast}) \\
&= \alpha_{k+1} \nabla f(x_k)^\top(x_k - z_k) + \alpha_{k+1} \nabla f(x_k)^\top(z_k - x_{\ast})\\
&=  \frac{\alpha_{k+1}(1-\tau_k)}{\tau_k} \nabla  f(x_k)^\top(y_k - x_k) + \alpha_{k+1} \nabla f(x_k)^\top(z_k - x_{\ast})\\
&\le  \frac{\alpha_{k+1}(1-\tau_k)}{\tau_k} (f(y_k) - f(x_k)) + \alpha_{k+1}^2L (f(x_k) - f(y_{k+1}))   + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 \\ 
&= (\frac{\alpha_{k+1}}{\tau_k} - \alpha_{k+1})(f(y_k) - f(x_k)) + \alpha_{k+1}^2L (f(x_k) - f(y_{k+1}))   + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 \\  
&=(\alpha_{k+1}^2L -\alpha_{k+1}) f(y_k) - \alpha_{k+1}^2L f(y_{k+1}) + \alpha_{k+1} f(x_k) + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2  - \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 ,\text{Let } \tau_k = \frac{1}{\alpha_{k+1} L}\\  
\end{align}
$$


为了希望利用上述不等式持续递推，取


$$
\begin{align}
\text{Set } \alpha_k &= \frac{k+1}{2L} , \\
\text{Then } \alpha_k^2L &= \alpha_{k+1}^2 L - \alpha_{k+1} + \frac{1}{4L}
\end{align}
$$


因此，


$$
\begin{align}
\alpha_{k+1}^2 L f(y_{k+1}) - \alpha_k^2 L f(y_k)  +\frac{1}{4L} f(y_k) + \frac{1}{2} \Vert z_{k+1} - x_{\ast} \Vert^2 - \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert^2 \le \alpha_{k+1} f(x_{\ast}) 
\end{align}
$$


递推则得到了，


$$
\begin{align}
\alpha_k^2 L f(y_k) - \alpha_0^2L f(y_0) + \frac{1}{4L} \sum_{i=0}^{k-1} f(y_k) + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2 &\le \sum_{i=0}^{k-1} \alpha_k f(x_{\ast}) \\
\alpha_k^2 L f(y_k) + \frac{1}{4L} \sum_{i=1}^{k-1} f(y_k) + \frac{1}{2} \Vert z_{k} - x_{\ast} \Vert - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2 &\le \sum_{i=0}^{k-1} \alpha_k f(x_{\ast}) \\
\alpha_k^2 L f(y_k) + \frac{1}{4L} \sum_{i=1}^{k-1} f(y_k)  - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2 &\le \sum_{i=0}^{k-1} \alpha_k f(x_{\ast}) \\
\alpha_k^2 L f(y_k) + \frac{1}{4L} \sum_{i=1}^{k-1} f(y_k)  - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2 &\le \sum_{i=0}^{k-1} \alpha_k f(x_{\ast}) \\
\frac{(k+1)^2}{4L} f(y_k) + \frac{k}{4L} f(x_{\ast}) - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2  &\le \frac{k(k+2)}{4L} f(x_{\ast}) \\
\frac{(k+1)^2}{4L} f(y_k) - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2  &\le \frac{k(k+1)}{4L} f(x_{\ast}) \\
\frac{(k+1)^2}{4L} f(y_k) - \frac{1}{2} \Vert z_0 - x_{\ast} \Vert^2  &\le \frac{(k+1)^2}{4L} f(x_{\ast}) \\ 
f(y_k) - f(x_{\ast}) &\le \frac{2L\Vert z_0 -x_{\ast} \Vert^2}{(k+1)^2}
\end{align}
$$


经过Linear Coupling之后1算法的收敛率达到了 $\mathcal{O}(\frac{1}{k^2})$, 这与该问题上的最优算法的复杂度相匹配，

虽然上述的证明仅仅针对于2范数，但可以使用相同的过程推广到其他范数上， 