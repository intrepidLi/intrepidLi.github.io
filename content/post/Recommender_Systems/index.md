---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "Introduction to Recommender Systems"
subtitle: ""
summary: ""
authors: []
tags: ["Recommender Systems"]
categories: ["Recommender Systems"]
date: 2025-12-11T13:34:27+08:00
lastmod: 2025-12-11T13:34:27+08:00
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

基于 UMN CSCI5123 和 FunRec 推荐系统整理
<!-- # Introduction to Recommendation Systems -->

## Recommender Tasks

2 key tasks:
- Predict how much a user will like an item
- Recommend items a user might like(推荐一个概率)

## Scoring items

从数学上来看，可以把 prediction 当成一个 scoring function:

$$
s(i;u)
$$

就是给 user $u$ 计算一个 item $i$ 的 score.

### Expanding Scoring
What about

- Current context (at the theater, 11:30 AM on the streetcorner)
- Query terms (search)

Question often arises: how does **search relate to recommendation**?

### Full Scoring Function

$s(i; u, q, x)$

- $i$: the item to score
- $u$: the active user
- $q$: the user's query
- $x$: the current context

Different systems use different variables

$s(i; u)$: traditional recommender: 只给 user 推荐一个 item分数
$s(i; q)$: traditional search： 只根据 query 找到最相关的 item 
$s(i; u, q)$: personalized search：根据 user 和 query 找到最相关的 item
$s(i; u, x)$: context-aware recommender：根据 user 和 context （场景或者环境）推荐 item
$s(i;u, q, x)$: context-aware pers. search
(Google, Bing)

### Computing $s$

- Much of what we do is compute $s$
- Content-based filtering: compute from user taste profile
- Demographic: compute from user demographics + segmented preferences
- Association rules: compute from context of currently-displayed item
- Collaborative: compute from user preferences and community preferences

### Scoring to Recommendation
Likewise, we can define an ordering function to produce recommendation lists:
$$O(I; u, q, x)$$
Like $s$, but takes a set of items $I$ and orders them instead of scoring a single item.

### Basic Top-N Recommendation

$O(I; u, q, x)$ is defined by:

- Score each item $i \in I$ using $s(i; u, q, x)$
- Sort items in decreasing order of score
- Truncate after n items

### Tweaking Top-N Recommendation

Variations of $O(I; u, q, x)$ may choose other orders:
- Diversity top-N to avoid too much similarity
- Re-prioritize top-N to promote high-value items
- 比如有另一套系统计算 item 的商业价值 score, 然后把这个 score 和推荐系统的 score 结合起来，得到一个新的 score 来排序.

### Extended Recommendation

An ordering may also depend on the number of items desired:

$$O(n, l; u, q, x)$$

Some recommenders may produce different top-5 and top-10 lists!


评分就是 $s$，排序就是 $O$。

所有的 [Notation](/uploads/Common_Notation.pdf) 都在这里。

# NN Collaborative Filtering

Collaborative Filetering(CF) 是一类 Recommendation: 要点是只考虑 interactions between users and items, 而不考虑其他信息(比如 user是谁，item里面有什么)。




<!-- **Key Concepts**

- Nearest Neighbor Collaborative Filtering
- User-User CF Algorithm
  - Neighborhoods and Tuning Parameters
  - Alternatives to Historic Agreement (social, trust)
- Item-Item CF Algorithm
  - Dealing with Unary Data
  - Hybrids and Extensions
  - Practical Implications
- Advanced Topics
  - Cold Start, Groups, Explanations, Threats -->

## User-User Collaborative Filtering

想做 非 personalized recommendation:

算一个score:

$$
s (u,i) = \frac{\sum_{v \in U} r_{vi}}{\left\vert U \right\vert } \tag{1}
$$

where $s$ is prediction score, $u$ is the active user, $i$ is the item to score, $U$ is the set of users who have rated item $i$, and $r_{vi}$ is the rating given by user $u$ to item $i$.

转化成：

$$
s(u, i) = \frac{\sum_{v \in U} r_{vi} \cdot w_{uv}}{ \sum_{ v \in  U} w_{uv} } \tag{2}
$$

这里的 $w_{uv}$ 是 user $u$ 和 user $v$ 之间的相似度（similarity）

问题：人们之间的 rate 可能相差很大

解决：

$(1)$ 变成 $(3)$:

$$
s(u, i) = \bar{r_{u}} + \frac{\sum_{v \in U} (r_{vi} - \bar{r_v}) }{ \left\vert U \right\vert  } \tag{3}
$$

$(2)$ 变成 $(4)$:

$$
s(u, i) = \bar{r_{u}} + \frac{\sum_{v \in U} (r_{vi} - \bar{r_v}) \cdot w_{uv}}{ \sum_{ v \in  U} w_{uv} } \tag{4}
$$

这样做的目的：

Because users may rate on different scales, and this adjustment makes it possible to **mix together ratings** from high-biased and low-biased users.


$v \in U$ 计算所有用户存在问题：

- 计算量大
- 可能有噪声用户（负相关性）

Common Characteristics

- Collection of Ratings
- Measure of Inter-User Agreement
  - Correlation, Vector Cosine
- Personalized Recommendations/Predictions
  - Weighted Combinations of Others' Ratings
- Tweaks to make things work right ...
  - Neighborhood limitations
  - Normalization
  - Dealing with limited co-ratings

形式化一下：

Given a set of items $I$, and a set of users $U$, and a sparse matrix of ratings $R$,

We compute the prediction $s(u, i)$ as follows:
- For all users $v \neq u$, compute $w_{uv}$
  - similarity metric (e.g., Pearson correlation)
- Select a neighborhood of users $V \subset U$ with highest $w_{uv}$
  - may limit neighborhood to top-k neighbors
  - may limit neighborhood to sim > sim\_threshold
  - may use sim or |sim| (risks of **negative correlations**)
  - may limit neighborhood to people who rated $i$ (if single-use)
- Compute prediction:

$$
s(u, i) = \bar{r_u} + \frac{\sum_{v \in V} (r_{vi} - \bar{r_v}) \cdot w_{uv}}{ \sum_{ v \in  V} w_{uv} } \tag{5}
$$

注意，这里的normalization其实不像统计一样把 均值归一化到0，标准差归一化到1，而是把用户的评分都变成相对于该用户的平均评分的偏差，因为practice更有效。

### Implementation Issues

- Given $m = \left\vert U \right\vert $ users and $n = \left\vert I \right\vert $ items:
  - Computation can be a Bottleneck
  - Correlation between two users is $O(n)$
  - All correlations for a user is $O(mn)$
  - All pairwise correlations is $O(m^2n)$
  - Recommendations at least $O(mn)$
- Lots of ways to make more practical
  - More persistent neighborhoods (m->k)
  - Cached or incremental correlations

注意，这个里面的neighbourhood 不是通过聚类得到的，而是 对每个用户，都有一个 neighbourhood，这个用户在这个 neighbourhood的center

### Core Assumptions/Limitations
- Why does this work?
  - Let's break it down ...
- Assumption: Our past agreement predicts our future agreement
  - Base Assumption \#1: Our tastes are either **individually stable** or move in sync with each other
  - Base Assumption \#2: Our system is scoped within a domain of agreement: 只有在某个领域内，用户的偏好才有意义比如电影推荐给电影爱好者，跨域的推荐效果不好

### Configure User-User CF

#### Selecting Neighborhoods

- All the neighbors
- Threshold similarity or distance
- Random neighbors
- Top-N neighbors by similarity or distance
- Neighbors in a cluster

#### How Many Neighbors?
- In theory, the more the better
- If you have a good similarity metric
- In practice, **noise from dissimilar neighbors decreases usefulness**
- Between 25 and 100 is often used
- 30-50 often good for movies

#### Scoring from Neighborhoods
- Average
- Weighted average
- Multiple linear regression
- Weighted average is common, simple, and works well

#### Common Normalizations
- Subtract user mean rating
- Convert to **z-score** (1 = 1 standard deviation above mean)
- Subtract item or item-user mean
Must reverse normalization after computing

$$
s(u, i) = \bar{r_u} + \frac{\sum_{v \in V} (r_{vi} - \bar{r_v}) \cdot w_{uv}}{ \sum_{ v \in  V} |w_{uv}| } 
$$

反向 norm就是 把上面的式子右半部分加的 $\bar{r_u}$.

#### Computing Similarities

如何评估相关性 $w_{uv}$?

Pearson Correlation:

$$
w_{uv} = \frac{\sum_{i \in I}(r_{ui } - \bar{r_{u}})(r_{vi} - \bar{r_v})}{ \sigma_{u} \sigma_{v} } \tag{6}
$$

这里，$I$ 是 user $u$ 和 user $v$ 都评分过的 items 的集合，$\sigma_{u}$ 和 $\sigma_{v}$ 分别是 user $u$ 和 user $v$ 的评分的标准差。

- only over ratings in common
- Spearman rank correlation 是 Pearson applied to **ranks**

局限： 

- small overlap
- 数据太小, 两个用户只有 一个或两个 的评分时，没法判断是否相似
- 考虑这样一个场景：两个用户有很多评分，但是只有一个评分是相同的，那么这个评分的相似度就会很高，但是实际上这两个用户并不相似。

#### Good Baseline Configuration
- Top N neighbors (~30)
- Weighted averaging
- User-mean or z-score normalization
- Cosine similarity over normalized ratings

## Item-Item Collaborative Filtering

### Motivation

来源于 User-User CF 的一些问题：

- Issues of Sparsity
  - With large item sets, small numbers of ratings, too often there are points where no recommendation can be made (for a user, for an item to a set of users, etc.) 单个用户的评分太少，很难找相似
  - Many solutions proposed here, including "filterbots", item-item, and dimensionality reduction
- Computational Performance($O(mn)$, $m$ 是用户数，$n$是物品数)
  - With millions of users (or more), computing all-pairs correlations is expensive
  - Even incremental approaches were expensive
  - And user profiles could change quickly - needed to compute **in real time** to keep users happy

### Core Insight

要求 users 要多于 items:

- Item-Item similarity is **fairly stable** 
  - This is dependent on having many more users than items
    - Average item has many more ratings than an average user
    - Intuitively, items don't generally change rapidly - at least not in ratings space (special case for time-bound items)
- Item similarity is a route to computing a prediction of a user's item preference


**Two step process:**
- Compute similarity between pairs of items
• Correlation between rating vectors
- co-rated cases only (only useful for multi-level ratings)
• Cosine of item rating vectors
- can be used with multi-level or unary ratings
- or adjusted ratings (normalize before computing cosine)
• Some use conditional probability (unary)
- Predict user-item rating
• Weighted sum of rated "item-neighbors"
• Linear regression to estimate rating

**Item-Item Top-N**
- Item-Item similarity model can be used to compute top-N directly:
  - Simplify model by limiting items to small "neighborhoods" of **$k$ most-similar items** (e.g., 20)
  - For a profile set of items, compute/merge/sort the $k$-most similar items for each profile item
    - Straightforward matrix operation from Deshpande and Karypis 

### Benefits of Item-Item
- It actually works quite well
  - Good prediction accuracy
  - Good performance on top-N predictions
- Efficient implementation
  - At least in cases where $\left\vert U \right\vert \gg \left\vert I \right\vert$ 用户数量大于物品数量
  - Benefits of precomputability
- Broad applicability and flexibility
  - As easy to apply to a shopping cart as to a user profile

### Core Assumptions/Limitations

- Item-item relationships need to be stable 
  - Mostly a corollary of stable user preferences
  - Could have special cases that are difficult (e.g., calendars, short-lived books, etc.)
  - Many of **these issues are general temporal issues** 时间问题是所有推荐系统的通病
- Main limitation/complaint: lower serendipity(意外发现)
  - This is a user/researcher complaint, not fully studied; intuition is clear

### Item-Item CF Algorithm

**Structure:**

- Pre-compute item similarities over all pairs of items
- Look for items similar to those the user Likes 
  - Or has purchased
  - Or has in their basket/cart

$$
s(u, i) = \frac{\sum_{v} w_{uv} f_{vi}}{\sum_{v} |w_{uv}|} \tag{7}
$$

$$
s(u,i) = \frac{\sum_{j \in N} w_{ij} \hat{r_{uj}}}{\sum_{j}|w_{ij}|}  \tag{8}
$$

怎么算 $w_{ij} = \text{sim}(i, j)$:

$$
\text{sim}(i, j) = \cos (\hat{r_i}, \hat{r_j}) = \frac{\hat{r_i} \cdot \hat{r_j}}{ \left\| \hat{r_i} \right\| \left\| \hat{r_j} \right\| } = \frac{\sum_{u} \hat{r_{ui}} \hat{r_{uj}}}{\sqrt{\sum_{u} \hat{r_{ui}}^2} \sqrt{\sum_{u} \hat{r_{uj}}^2}}  = \frac{\sum_{u}(r_{ui} - \bar{r_i})(r_{uj} - \bar{r_j})}{\sqrt{\sum_{u} (r_{ui} - \bar{r_i})^2} \sqrt{\sum_{u} (r_{uj} - \bar{r_j})^2}} \tag{9}
$$

这最后一个式子就是 Pearson correlation.

What are the benefits of **using a cosine similarity between mean-centered** (normalized by subtracting the item mean) item rating vectors?

- When we treat missing values as 0, it introduces a useful self-damping effect when items have few users in common but many ratings.

- It is equivalent to the Pearson correlation, so it is statistically meaningful.

# 编排

- 2.召回模型
  - 2.1 协同过滤
  - 2.2 向量召回
  - 2.3 序列召回
    - 2.3.1 深化用户兴趣表示
    - 2.3.2 生成式召回方法
    - 2.3.3 总结

### 序列召回

之前的 **协同过滤** 和 **向量召回** 通常将用户的历史行为汇总成一个静态的表示（比如一个向量），然后基于这个表示进行推荐

用户的行为是有时间顺序的，所以 **序列召回** 是利用 **用户行为的时间顺序信息** 进行推荐。

**Core Insight:** User 的 当前兴趣不仅取决于过去喜欢什么，还取决于最近在做什么，以及这些行为的顺序。


两类具有代表性的方法：

- 多兴趣用户表示
  - 传统方法**将用户压缩为单一向量**，难以充分表达用户兴趣的**多样性和时序性**。
  - 多兴趣表示方法尝试通过**多个向量或动态结构**来更好地刻画用户的复杂兴趣模式。

- 生成式序列预测
  - 将推荐问题重新定义为序列生成任务，借助NLP中的生成模型（如Transformer）来预测用户的下一个行为。
