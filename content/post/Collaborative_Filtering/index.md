---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "Collaborative Filtering"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Recommender Systems"]
categories: ["Recommender Systems"]
date: 2025-12-12T20:09:58+08:00
lastmod: 2025-12-12T20:09:58+08:00
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

# ItemCF

Collaborative Filtering based on items.

基本假设：用户兴趣有一定连贯性，用户喜欢物品 A, 那么用户也可能喜欢与物品 A 相似的物品 B。

![itemcf](images/itemcf_illustration.svg)

给左边用户推荐某个商品：先分析 T-shirt 和 夹克之间的相似度，因为右边三个用户同时喜欢 T-shirt 和 夹克，所以认为 T-shirt 和 夹克相似度较高，那么给左边用户推荐夹克的可能性就比较大。

## 算法

先量化物品之间的相似度：在实际场景中一般只有用户和物品的交互数据（点击、购买、收藏等），不会有评分数据：

一种naive的想法：每个物品表示成一个用户向量，计算向量相似度，**缺点：** 商品数量巨大时，计算所有物品对相似度时间复杂度是 $O(|I|^{2})$

但是因为 稀疏化特点，很多物品对之间没有共同用户，所以可以只计算有共同用户的物品对相似度：

- 用户-物品倒排表：每个用户维护一个交互过的物品列表
- 计算物品共现矩阵：创建一个矩阵 `C[i][j]` 记录 物品 i 和 j 的共同用户数量。遍历所有用户物品列表，把列表中物品两两配对，对应 的 `C[i][j]` 加一，构造共现矩阵。
- 计算最终相似度：余弦相似度计算物品相似度：
  $$
  w_{ij} = \frac{C[i][j]}{\sqrt{|N(i)| \cdot |N(j)|}}
  $$
  其中 $|N(i)|$ 表示与物品 i 有交互的用户集合, `C[i][j]` 表示物品 i 和 j 的共同用户数量。分母是一个归一化的过程，防止热门物品对相似度的影响过大（热门商品 i 的 `C[i][j]` 大，但是同时 $|N(i)|$ 也大）。

算法复杂度：

- 暴力：