# k-means 聚类中使用余弦距离 cos distance

> 本文在 [知乎](https://zhuanlan.zhihu.com/p/380389927) 发布

k-means 聚类算法中使用欧氏距离作为判别标准，本文讨论使用余弦距离作为判别的方法和理论基础。

先说结论：**使用欧氏距离聚类结果等价于使用余弦距离聚类结果**。

首先看余弦的计算 $\forall x_j,x_k \in\mathbb{R}^m$,

$$
\operatorname{cos} \theta = \frac{x_j}{||x_j||}\cdot \frac{x_k}{||x_k||}
$$

可以看做两个归一化后的单位向量的内积，同时理解对样本向量进行归一化并不改变余弦距离的计算。

而在欧氏距离的计算中，

$$
||x_j - x_k||^2 = x_j^Tx_j - 2 x_j^Tx_k + x_k^Tx_k
$$

如果向量已经是单位向量，那么

$$
||x_j - x_k||^2 = 2(1 - x_j^Tx_k)
$$

也即使用余弦距离和使用欧氏距离进行判别的聚类结果是等价。

所以使用余弦距离进行聚类的方式是先将样本进行归一化，然后使用常规方法和工具（如 scikit-learn）进行计算。

余下的问题则是余弦聚类的中心和距离怎么计算？

对于维度为 m 的 n 个样本

$$
x_1, x_2, \dots x_n \in\mathbb{R}^m
$$

求样本余弦中心 $\hat x \in \mathbb{R}^m $ 使得

$$
\operatorname{min} \sum_{k=1}^n |\operatorname{cos}\theta_k|
$$

其中 

$$
\operatorname{cos} \theta_k = \frac{x_k\cdot \hat x}{||x_k||\cdot||\hat x||}.
$$

把问题等价重写为目标

$$
\operatorname{min} -\sum_{k=1}^n \frac{x_k\cdot \hat x}{||x_k||\cdot||\hat x||} = \operatorname{min} -\frac{1}{c}\sum_{k=1}^n \frac{1}{||x_k||} \left(\sum_{l=1}^m x_{kl}\cdot \hat x_l\right)
$$

和约束

$$
\sum_{l=1}^{m} (\hat x_l)^2 = c^2, \quad c\in\mathbb{R}, c>0.
$$

则使用 Lagrange multiplier 方法有，

$$
\mathcal{L}(\hat x_1,\hat x_2,\dots,\hat x_m,\lambda)  = -\frac{1}{c}\sum_{k=1}^n \frac{1}{||x_k||} \left(\sum_{l=1}^m x_{kl}\cdot \hat x_l\right) +\lambda\left( \sum_{l=1}^{m} (\hat x_l)^2 - c^2 \right)
$$

于是 $\forall \hat x_l$， 令

$$
\frac{\partial \mathcal{L}(\hat x_1,\hat x_2,\dots,\hat x_m,\lambda)}{\partial \hat x_l} = = -\frac{1}{c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||}   +2\lambda \hat x_l = 0
$$

推出

$$
\hat x_l = \frac{1}{2\lambda c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||}
$$

带回约束条件中，由

$$
\sum_{l=1}^{m} \left( \frac{1}{2\lambda c}\sum_{k=1}^n \frac{x_{kl}}{||x_k||}  \right)^2 = c^2
$$

得到

$$
2\lambda c = \frac{1}{c} \left( \sum_{l=1}^{m}  \left( \sum_{k=1}^n \frac{x_{kl}}{||x_k||}  \right)^2  \right)^{1/2}
$$

于是

$$
\hat x_l = c\cdot  \left( \sum_{k=1}^n \frac{x_{kl}}{||x_k||}  \right) \left( \sum_{l=1}^{m}  \left( \sum_{k=1}^n \frac{x_{kl}}{||x_k||}  \right)^2  \right)^{-1/2}
$$

至此，就得到了余弦中心的计算方法。

注意到这个余弦中心向量的方向和模长并没有关系，这也和余弦距离的特性相符合。

当对样本进行归一化，同时假定余弦中心向量归一化后,

$$
\hat x_l =  \left( \sum_{k=1}^n {x_{kl}}  \right) \left( \sum_{l=1}^{m}  \left( \sum_{k=1}^n {x_{kl}}  \right)^2  \right)^{-1/2} = \bar x_l \left( \sum_{l=1}^{m}  {\bar x_l}^2  \right)^{-1/2} = \frac{\bar x_l}{||\bar x||}
$$


即余弦中心为欧氏中心归一化后的结果

$$
\hat x  = \frac{\bar x}{||\bar x||}
$$

附：使用 scikit-learn 进行计算的代码

```python
# 归一化
nm = np.sqrt((X**2).sum(axis=1))[:,None]
X = X / nm

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

# 其实也是在计算归一化
mm = np.sqrt(np.square(kmeans.cluster_centers_).sum(axis=1)[:,None])
cos_centers = kmeans.cluster_centers_ / mm

distance = 1 - np.dot(cos_centers, X.T)
```
