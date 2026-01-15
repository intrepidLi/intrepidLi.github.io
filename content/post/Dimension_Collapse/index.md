---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "Dimension Collapse"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Recommender Systems", "Summary"]
categories: ["Recommender Systems", "Summary"]
date: 2026-01-05T17:09:27+08:00
lastmod: 2026-01-05T17:09:27+08:00
featured: false
draft: true

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

# Preliminaries

## Low-Cardinality Fields

两个集合：

- Field $A$，基数(cardinality $n_{A}$)
  - 比如 表示用户性别，只有 `男` 和 `女` 两个取值，那么基数 $n_{A}=2$
- Field $B$，基数(cardinality $n_{B}$)

$n_{A} \gg n_{B}$

$V_{A} \in \mathbb{R}^{n_{A} \times d}$ 和 $V_{B} \in R_n^{n_{B} \times d}$ collect row embeddings $\{ v_i \}_{i \in A}$ and $\{ v_j \}_{j \in B}$



一个比较好的类比是 NLP中的Word Embedding:

- Field $A$ 类似于一个大词汇表 (large vocabulary)
- Cardinality $n_{A}$ 类似于词汇表大小 (vocabulary size)
- Index $i \in A$ 就是 token ID 
- Embedding $v_i$ 就是 token 的 embedding，是矩阵 $V_{A}$ 第 $i$ 行的转置 ($d \times 1$)
- Matrix $V_{A}$ 就是 embedding 矩阵

每个 training sample 是一个 triplet $(i, j, y)$: $i \in A, j \in B$, label $y \in \{ 0, 1 \}$

分数：

$$
s(i, j) = <\boldsymbol{v_i}, \boldsymbol{v_j}> = \boldsymbol{v_i}^T \boldsymbol{v_j} \in \mathbb{R}, \hat{y}(i, j) = \sigma(s(i, j))
$$, $\sigma(t) = \frac{1}{1 + e^{-t}}$

标准化的 dataset $\mathcal{D}$ 上的BCE loss是：

$$
\begin{aligned}
\mathcal{L}(V_{A}, V_{B}) &= \frac{1}{|\mathcal{D}|} \sum_{(i, j, y) \in \mathcal{D}} \left[ -y \log \hat{y}(i, j) - (1 - y) \log (1 - \hat{y}(i, j)) \right] + \frac{\lambda}{2}\| V_{A} \|_F^2 + \frac{\lambda}{2} \| V_{B} \|_F^2 \\
&= \frac{1}{|\mathcal{D}|} \boldsymbol{e_{A}}^{\top} \left[ (\boldsymbol{e_{A}}\boldsymbol{e_{B}}^{\top} - Y) \odot S - \log \hat{Y}\right]\boldsymbol{e_{B}} + \frac{\lambda}{2} \operatorname{tr}(V_{A}^{\top}V_{A}) + \frac{\lambda}{2} \operatorname{tr}(V_{B}^{\top}V_{B}) \\
&= \frac{1}{|\mathcal{D}|} \boldsymbol{e_{A}}^{\top} \left[ (\boldsymbol{e_{A}}\boldsymbol{e_{B}}^{\top} - Y) \odot (V_{A}V_{B}^{\top}) - \log \sigma(V_{A}V_{B}^{\top})\right]\boldsymbol{e_{B}} + \frac{\lambda}{2} \operatorname{tr}(V_{A}^{\top}V_{A}) + \frac{\lambda}{2} \operatorname{tr}(V_{B}^{\top}V_{B})
\end{aligned}
$$

$$
d \mathcal{L} = \operatorname{tr}(\frac{\partial \mathcal{L}}{\partial V_{A}}^{\top} d V_{A})
$$

$$
\begin{aligned}
d \mathcal{L} &= \frac{1}{|\mathcal{D}|} \boldsymbol{e_{A}}^{\top} d \left[ (\boldsymbol{e_{A}}\boldsymbol{e_{B}}^{\top} - Y) \odot (V_{A}V_{B}^{\top}) - \log \sigma(V_{A}V_{B}^{\top}) \right] \boldsymbol{e_{B}} + \frac{\lambda}{2} \operatorname{tr}(d V_{A}^{\top} V_{A} + V_{A}^{\top} d V_{A}) \\
&= \frac{1}{|\mathcal{D}|} \operatorname{tr} (\boldsymbol{e_{B}} \boldsymbol{e_{A}}^{\top} d \left[ (\boldsymbol{e_{A}}\boldsymbol{e_{B}}^{\top} - Y) \odot (V_{A}V_{B}^{\top}) - \log \sigma(V_{A}V_{B}^{\top}) \right]) + \lambda \operatorname{tr}(V_{A}^{\top} d V_{A}) \\
&= \frac{1}{|\mathcal{D}|} \operatorname{tr} (\boldsymbol{e_{B}} \boldsymbol{e_{A}}^{\top} [(\boldsymbol{e_{A}}\boldsymbol{e_{B}}^{\top}) \odot (dV_{A}V_{B}^{\top})]) - \frac{1}{|\mathcal{D}|} \operatorname{tr} (\boldsymbol{e_{B}} \boldsymbol{e_{A}}^{\top}  [Y \odot (dV_{A}V_{B}^{\top})]) - \frac{1}{|\mathcal{D}|} \operatorname{tr}(\boldsymbol{e_{B}} \boldsymbol{e_{A}}^{\top} d \log \sigma(V_{A} V_{B}^{\top})) + \operatorname{tr}(\lambda V_{A}^{\top} d V_{A}) \\
&= \frac{1}{|\mathcal{D}|} \operatorname{tr} ([\boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top} \odot \boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top}]^{\top} d V_{A} V_{B}^{\top}) - \frac{1}{|\mathcal{D}|} \operatorname{tr} ([Y \odot \boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top}]^{\top} d V_{A} V_{B}^{\top}) - \frac{1}{|\mathcal{D}|} \operatorname{tr} (\boldsymbol{e_{B}} \boldsymbol{e_{A}}^{\top}(\boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top} - \sigma(V_{A} V_{B}^{\top})) \odot d V_{A} V_{B}^{\top} ) + \operatorname{tr}(\lambda V_{A}^{\top} d V_{A}) \\
&= \frac{1}{|\mathcal{D}|} \operatorname{tr}(V_{B}^{\top}(\boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top})^{\top} d V_{A}) - \frac{1}{|\mathcal{D}|} \operatorname{tr}(V_{B}^{\top} (Y)^{\top} d V_{A}) - \frac{1}{|\mathcal{D}|} \operatorname{tr}( V_{B}^{\top}(\boldsymbol{e_{A}} \boldsymbol{e_{B}}^{\top})^{\top} dV_{A}) - \frac{1}{|\mathcal{D}|} \operatorname{tr}(- V_{B}^{\top}\sigma (V_{A}V_{B}^{\top})^{\top}dV_{A}) + \operatorname{tr}(\lambda V_{A}^{\top} d V_{A}) \\
\end{aligned}
$$

$$
\frac{\partial \mathcal{L}}{\partial V_{A}} = \frac{1}{|\mathcal{D}|} ( - YV_{B} + \sigma(V_{A}V_{B}^{\top})V_{B}) + \lambda V_{A} = \frac{1}{|\mathcal{D}|} (\sigma(V_{A}V_{B}^{\top}) - Y)V_{B} + \lambda V_{A}
$$

注意这个梯度不会因为 $|\mathcal{D}|$ 变大而变小，因为 $\sigma - Y$ 这个东西只有在 $(i, j)$ 属于 dataset $\mathcal{D}$ 时才不为0，所以实际上 $\frac{1}{|\mathcal{D}|} \sum_{(i, j) \in \mathcal{D}} (\sigma(s(i, j)) - y_{ij}) v_{j}$ 这个和的项数是和 $|\mathcal{D}|$ 成正比的。

$\boldsymbol{e_{B}} \in \mathbb{R}^{n_{B} \times 1} = \text{all } 1$, $\boldsymbol{e_{A}} \in \mathbb{R}^{n_{A} \times 1} = \text{all } 1$

根据对称性，取一个转置和对换角标（原因：Loss函数中 $V_{A}$ 和 $V_{B}$ 的关系：$S = V_{A} V_{B}^{\top} $，要取 $S^{\top} = V_{B} V_{A}^{\top}$）所以：

$$
\frac{\partial \mathcal{L}}{\partial V_{B}} = \frac{1}{|\mathcal{D}|} (\sigma(V_{A}V_{B}^{\top}) - Y)^{\top} V_{A} + \lambda V_{B}
$$

**Theorem 1**

Stationary-point span 相等：

原因：设 $V_{A}^{*}, V_{B}^{*}$ 是loss的一个stationary point:

$$
\frac{\partial \mathcal{L}}{\partial V_{A}^{*}} = 0, \frac{\partial \mathcal{L}}{\partial V_{B}^{*}} = 0
$$

则：

$$
\lambda V_{A}^{*} = - \frac{1}{|\mathcal{D}|} (\sigma(V_{A}^{*}V_{B}^{* \top}) - Y)V_{B}^{*}
$$

$$
\lambda V_{B}^{*} = - \frac{1}{|\mathcal{D}|} (\sigma(V_{A}^{*}V_{B}^{* \top}) - Y)^{\top} V_{A}^{*}
$$

$span$ 的定义是 $Y = AX$，则 $Y \subseteq span(A)$, 原因：$Y$ 是 $A$ 的列的线性组合。
 
注意 Theorem 1说的 $span$ 其实是 Row span 相等，因为可以看出 $V_{A}^{*}$ 和 $V_{B}^{*}$ 互属于对方的 Row span。

之后可以推出：

$$
\operatorname{dim}(span(V_{A}^{*})) = \operatorname{dim}(span(V_{B}^{*})) \implies rank(V_{A}^{*}) = rank(V_{B}^{*})
$$

注意，$\operatorname{rank}$ 的意义就是 矩阵的 Row span 或 Column span 的dim。

可以看出 $\operatorname{rank}(V_{A}^{*}) = \operatorname{rank}(V_{B}^{*}) \le \min \{ d, n_{A}, n_{B} \}$ 原因：矩阵的rank不可能超过矩阵的行数或列数。

所以如果 $n_{B}$ 特别小的话，比如 $n_{B} = 2$ (性别)，那么 $\operatorname{rank}(V_{B}^{*}) \le 2$，所以 $\operatorname{rank}(V_{A}^{*}) \le 2$，所以 $V_{A}^{*}$ 的 row span 最多只有2维。

所以比如 $V_{A}$ 的mini-Batch $\mathcal{B}_{t} \subset \mathcal{D}$ SGD就是：

$$
V_{A}^{(t+1)} = V_{A}^{(t)} - \eta_{t} \left( \frac{1}{|\mathcal{B}_{t}|} (\sigma(V_{A}^{(t)}V_{B}^{(t)\top}) - Y)V_{B}^{(t)} + \lambda V_{A}^{(t)} \right)
$$

$S_t = span(V_{B}^{(t)})$

## Frequency Skew

