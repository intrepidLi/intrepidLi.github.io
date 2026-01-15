---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "MemGen"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Papers", "latent_memory"]
categories: ["latent_memory"]
date: 2025-12-24T16:27:05+08:00
lastmod: 2025-12-24T16:27:05+08:00
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

# MemGen-2509.24704

解决现在推理过程中记忆和推理无法联系的问题：

用trigger和weaver机制判断在推理的某个位置插入KV Cache 记忆从而实现 latent space 推理计算和记忆的结合。