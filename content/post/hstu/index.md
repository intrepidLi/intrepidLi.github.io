---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "HSTU"
subtitle: ""
summary: ""
authors: [admin]
tags: []
categories: ["Passages"]
date: 2025-12-10T13:15:11+08:00
lastmod: 2025-12-10T13:15:11+08:00
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

## 解决的问题

Despite being trained on huge volume of data with thousands of features, most **Deep Learning Recommendation Models (DLRMs)** in industry **fail to scale with compute**.

DLRMs 的特点：使用heterogeneous features(异构特征), 比如numerical features -- counters and ratios, embeddings, and categorical features such as creator ids, user ids, etc.
