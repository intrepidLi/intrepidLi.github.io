---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "GE4Rec"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2025-12-10T13:15:32+08:00
lastmod: 2025-12-10T13:15:32+08:00
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

ICML 26 Tencent 

推荐系统中的 Click-Through Rate(CTR) prediction 是一个核心任务：预测用户点击某个物品的概率。

特征交叉

## 动机与要解决的问题 (Motivation & Problem)

核心痛点：目前的CTR模型大多遵循判别式范式（Discriminative Paradigm），即通过**显式的特征交互**（Feature Interaction）来处理原始的ID Embeddings，然后估算点击概率。

这种传统范式存在两个严重的内生问题：

- 嵌入维度坍塌 (Embedding Dimensional Collapse)：由于“交互坍塌理论”，特征嵌入往往分布在一个低维子空间中。这意味着虽然Embedding的维度设定很高（如16维或64维），但其中很多维度是无效的或没有携带信息的，导致表达能力浪费。
- 信息冗余 (Information Redundancy)：现有的交互模型倾向于学习到高度相关的表示，导致信息冗余。这违反了“冗余减少原则（Redundancy-reduction principle）”，即理想的特征表示应该是彼此之间相关性较低的。

**动机**：受到计算机视觉（如MAR, VAR）和NLP（如Next-token prediction）领域生成式模型的启发，作者希望重新思考CTR数据的“内在结构”。虽然CTR数据是多域分类数据（无序），但特征之间的共现关系（Co-occurrence）可以被视为一种内在结构。

因此，作者提出用生成式范式来替代判别式范式，通过“特征生成”来学习更好的特征表示。

## 方法 (Methods)

作者提出了监督特征生成（SFG）框架。这个框架不仅仅是一个新模型，而是一种通用的重构方法，可以应用于几乎所有的现有CTR模型（如DeepFM, DCN V2等）。

核心架构：SFG框架包含两个主要部分，遵循“全预测全（All-Predict-All）”的思路：

- 编码器 (Encoder)：
  - 作用： 将所有原始特征（$x_{source}$）映射到一个隐藏空间（Latent Space），为每个特征生成一个新的、更鲁棒的隐藏表示 7777。
  - 实现： 极其简单但有效。作者使用了一个逐域（field-wise）的单层非线性MLP。具体来说，就是将所有特征拼接后，通过一个非线性激活函数（如ReLU）和投影矩阵进行变换。
  - 设计原则： 这种非线性变换是缓解维度坍塌的关键 
 
- 解码器 (Decoder)：
  - 作用： 将隐藏空间的表示映射回原始空间，旨在根据隐藏表示“生成”出所有的原始特征（$x_{target}$）。
  - 实现： 解码器的具体实现其实就是对应了传统CTR模型中的特征交互函数（如FM的点积、DCN的交叉层等）。
- 关键创新点 
  -  监督损失 (Supervised Loss)：与传统的生成式推荐（通常使用自监督损失，如预测下一个token）不同，CTR任务中天然**存在监督信号（点击/未点击）**。作者直接使用**监督损失（如交叉熵损失）**来优化生成过程。
  -  原因： 如果使用自监督（如Mask掉一个特征预测它），会导致标签泄漏（Label Leakage），因为在CTR中所有特征通常都是已知的。使用点击标签作为监督信号，可以强制编码器学习与用户反馈最相关的特征共现信息 
 

工作流程总结：

输入原始特征 $\rightarrow$ Encoder生成新表示 $\rightarrow$ Decoder利用新表示进行交互/生成 $\rightarrow$ Classifier计算得分 $\rightarrow$ 使用$y_{sup}$计算Loss 

## 做的实验 (Experiments)

在两个广泛使用的大规模数据集 **Criteo** 和 **Avazu** 上进行实验，并进行了在线A/B测试 

### 实验结果

#### 性能提升 (RQ1)：

将SFG框架应用于FM, DeepFM, DCN V2等多种基线模型后，AUC和Logloss均获得一致性提升 例如，生成式的CrossNet甚至击败了判别式的DCN V2（更复杂的模型），证明了范式转换的有效性 
- 缩小差距： 生成式范式缩小了不同模型架构之间的性能差距，说明好的特征表示比复杂的交互结构更重要 

#### 缓解维度坍塌 (RQ2 - Collapse)：

分析方法： 对Embedding矩阵进行奇异值分解（SVD），观察奇异值分布。

结果： 判别式模型的奇异值在某一点后会断崖式下跌（意味着后续维度无效），而生成式模型的奇异值分布更平缓，表明Embedding空间被利用得更充分 

#### 减少信息冗余 (RQ2 - Redundancy)：

分析方法： 计算特征间的皮尔逊相关系数。

结果： 判别式模型（尤其是FM）的特征间相关性很高（冗余大），而生成式DCN V2的相关矩阵几乎为零，实现了高度的去相关（De-correlation），符合冗余减少原则。

#### 消融研究 (RQ3)：

- Source设计： 使用“所有特征”作为输入效果最好，尤其是对低基数（low-cardinality）特征帮助巨大 。

- Encoder设计： 必须包含非线性激活函数（Non-linear），否则无法缓解坍塌 。简单的单层MLP优于复杂的Transformer或Attention结构 

- Target设计： “全预测全（Predict All）”优于类似Masked Image Modeling（随机遮蔽）的方法 。

工业界落地：在腾讯广告平台的A/B测试中，主要场景的GMV提升了2.68%，这是一个巨大的商业价值提升。

### 最终结论 (Conclusion)

通过坚实的理论分析和实验证明：

范式转移： 将CTR预测从“判别式特征交互”转变为“监督特征生成”是可行且更优的 

解决本质问题： SFG框架有效地解决了原始ID Embeddings存在的维度坍塌和信息冗余问题，生成了质量更高的特征表示 

通用性与高效性： 该框架可以无缝集成到现有的各种CTR模型中，且计算开销增加非常微小（时间增加约3%，显存增加约1.5%）

一句话总结： 作者并没有发明复杂的网络结构，而是通过引入Encoder和生成式视角，重新“清洗”和生成了特征Embedding，从而在几乎所有主流CTR模型上实现了“免费”的性能提升。