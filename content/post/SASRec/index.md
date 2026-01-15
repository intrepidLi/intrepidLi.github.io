---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "SASRec"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2025-12-24T16:28:50+08:00
lastmod: 2025-12-24T16:28:50+08:00
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

## 数据

Movie-Lens `ml-1m` 数据集：

原始`ratings.dat` 的数据样式：`UserID::MovieID::Rating::Timestamp`

处理 `ratings.dat`，具体处理过程：

``` python
# 1st Step: 读取 ratings.dat 文件
data = pd.read_csv(
  input_file_path, # 不用管后缀
  sep='::',
  engine='python',  # 使用 python 引擎处理多字符分隔符
  names=['UserID', 'MovieID', 'Rating', 'Timestamp'] # 没有表头行，指定列名
)

# 2nd Step: 按 UserID 和 Timestamp 升序排序
data_sorted = data.sort_values(by=['UserID', 'Timestamp']).reset_index(drop=True)

# 3rd Step: 重映射 UserID 和 MovieID

# --- 用户ID重映射 ---
# factorize 会将唯一值转换为一个从 1 开始的整数数组
data_sorted['User_ID_Mapped'], unique_users = pd.factorize(data_sorted['UserID'])
data_sorted['User_ID_Mapped'] = data_sorted['User_ID_Mapped'] + 1

# --- 物品ID重映射 (MovieID) ---
data_sorted['Item_ID_Mapped'], unique_items = pd.factorize(data_sorted['MovieID'])
data_sorted['Item_ID_Mapped'] = data_sorted['Item_ID_Mapped'] + 1
```

得到的数据样式：

`UserID MovieID Rating Timestamp`

``` latex
1 1 4 978300019
1 2 5 978300055
1 3 4 978300055
1 4 5 978300055
1 5 3 978300103
1 6 5 978300172
1 7 4 978300275
1 8 5 978300719
```


放在 `ml-1m_processed.txt` 中，格式为 `UserID ItemID Rating Timestamp`，空格分隔。

### 预处理

先进行 `build_index` : 在 `utils.py` 中：

```python
def build_index(dataset_name):

    ui_mat = np.loadtxt('data/%s.txt' % dataset_name, dtype=np.int32)

    n_users = ui_mat[:, 0].max()
    n_items = ui_mat[:, 1].max()

    u2i_index = [[] for _ in range(n_users + 1)]
    i2u_index = [[] for _ in range(n_items + 1)]

    for ui_pair in ui_mat:
        u2i_index[ui_pair[0]].append(ui_pair[1])
        i2u_index[ui_pair[1]].append(ui_pair[0])

    print('n_users: %d, n_items: %d' % (n_users, n_items))
    return u2i_index, i2u_index
```

#### 解释

输出 users 的数量 `n_users=6040`, items 的数量 `n_items=3706`。

这一步得到两个索引 list `u2i_index`，记录每个用户对应的交互物品列表：

```
u2i_index[:3]:
[
  [], 
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
  [54, 55, 56, 57, 58, 59, 60, 10, 61, 62, 63, 64, 65, 9, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 18, 78, 79, 80, 81, 82, 83, 84, 21, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 30, 121, 122, 123, 124, 125, 36, 126, 127, 128, 129, 130, 17, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175]
]
```

`u2i_index[1]` 代表用户 1 交互过的物品列表 `[1, 2, 3, ..., 53]`。
`u2i_index[2]` 代表用户 2 交互过的物品列表 `[54, 55, 56, ..., 175]`。
以此类推。

`i2u_index` 同理，记录每个物品对应的交互用户列表：

```
u2i_index[-2:]:
[[5851], [5938]]
```

`i2u_index[3705]` 代表物品 3705 交互过的用户列表 `[5851]`。
`i2u_index[3706]` 代表物品 3706 交互过的用户列表 `[5938]`。

### 训练集/验证集/测试集划分

`data_partition` 函数：

``` python
# train/val/test data generation
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')[:2]
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 4:                          # To be rigorous, the training set needs at least two data points to learn
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]
```

#### 解释

注意第一步算 `usernum` 和 `itemnum` 的时候，到最后得到的都是数据中最大的用户ID和物品ID，即 `n_users=6040`, `n_items=3706`。

因为我之前做了 ID 重映射，所以这里的用户ID和物品ID是连续的，从 1 开始到最大值，没有跳号，但是这种写法实际上是适用于没有重映射的情况。

一般来说，都要规避 ID 跳号的问题（会导致Embedding巨大且稀疏，但实际理论上不会影响模型效果），重映射是一个常用的手段。

实际上可以直接用 第一步 `build_index` 得到的 `n_users` 和 `n_items`，之后用 `u2i_index` 来划分训练集/验证集/测试集。上面的 `User` 实际等同于 `u2i_index`， `key` 是用户ID，`value` 是该用户对应的物品ID列表。

之后根据 每个user交互过的 物品数量：
- 小于4个的直接全归到 `user_train` 里面
- 大于等于4个的，最后两个物品分别归到 `user_valid` 和 `user_test`，其余的都归到 `user_train`

注意：`user_train`, `user_valid`, `user_test` 都是 dict 格式，key 是用户ID，value 是该用户对应的物品ID列表。

`len(user_train) = 6040`, `key` 从 1 到 6040。

#### 平均交互物品数和batch数

计算 `user_train` 的平均交互物品数：

``` python
cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))
```

`num_batch` 的计算：

``` python
num_batch = (len(user_train) - 1) // args.batch_size + 1
```

`args.batch_size` 默认为 128。

### 采样

为什么需要采样？

构造有监督数据：在NLP任务中，句子序列自回归有自然的监督信号，而在推荐系统中，用户的历史交互序列并没有明确的监督信号，需要通过采样来构造训练数据。

大致意思：

用户的历史交互序列：$s=\{ s_1,s_2, \ldots s_n \}$

目标是预测下一个用户可能交互的物品 $s_{n+1}$。

训练序列：

``` python
for t in [1, ..., n-1]:
    输入:  [s1, ..., st]
    正例:  s_{t+1}
    负例:  j (j ∉ S)
```


`sample_function` 和 `WarpSampler` 类：

``` python
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample(uid):

        # uid = np.random.randint(1, usernum + 1)
        while len(user_train[uid]) <= 1: uid = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[uid][-1]
        idx = maxlen - 1

        ts = set(user_train[uid])
        for i in reversed(user_train[uid][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
            nxt = i
            idx -= 1
            if idx == -1: break

        return (uid, seq, pos, neg)

    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))
```

#### 解释

`user_train` 是一个 `dict`，类似：

``` python
user_train = {
  1: [3, 5, 9, 12],
  2: [4, 7],
  3: [2, 8, 10, 11, 15]
}
```

key 是 用户ID，value 是该用户对应的物品ID列表。

先看 `sample(uid)` 函数：

``` python
while len(user_train[uid]) <= 1:
    uid = np.random.randint(1, usernum + 1)
```

确保采样的用户至少有两个交互物品，否则无法构造训练数据: (history -> next item)。

``` python
seq = np.zeros([maxlen], dtype=np.int32)
pos = np.zeros([maxlen], dtype=np.int32)
neg = np.zeros([maxlen], dtype=np.int32)
nxt = user_train[uid][-1]
idx = maxlen - 1
```

初始化三个长度为 `maxlen` 的序列 `seq`, `pos`, `neg`:
- `seq`: 存储用户的历史交互物品序列
- `pos`: 存储每个位置的正例物品序列（下一个交互物品）
- `neg`: 存储每个位置的负例物品序列（未交互物品）

`nxt` 是下一个真实点击物品（物品序列中的最后一个）

`idx` 是最后一个位置的索引。


``` python
ts = set(user_train[uid])
for i in reversed(user_train[uid][:-1]):
    seq[idx] = i
    pos[idx] = nxt
    neg[idx] = random_neq(1, itemnum + 1, ts)          # Don't need "if nxt != 0"
    nxt = i
    idx -= 1
    if idx == -1: break

return (uid, seq, pos, neg)
```



``` python
# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
```

比如 `user_train[uid] = [3, 5, 9, 12]`，则：
- 初始化：`nxt=12`, `maxlen=4`, `idx=maxlen-1=3`
- 第一次循环：`i=9`，`idx=maxlen-1=3`, `seq[3]=9`, `pos[3]=12`, `neg[3]=随机未交互物品1`, `seq: [0, 0, 0, 9]`, `pos: [0, 0, 0, 12]`
- 更新：`nxt=9`, `idx=2`
- 第二次循环：`i=5`，`seq[maxlen-2]=5`, `pos[maxlen-2]=9`, `neg[maxlen-2]=随机未交互物品2`, `seq: [0, 0, 5, 9]`, `pos: [0, 0, 9, 12]`
- 更新：`nxt=5`, `idx=1`
- 第三次循环：`i=3`，`seq[maxlen-3]=3`, `pos[maxlen-3]=5`, `neg[maxlen-3]=随机未交互物品3`, `seq: [0, 3, 5, 9]`, `pos: [0, 5, 9, 12]`
- 更新：`nxt=3`, `idx=0`
- 结束循环，返回 `(uid, seq, pos, neg)`:
- `seq: [0, 3, 5, 9]`, `pos: [0, 5, 9, 12]`, `neg: [0, 随机未交互物品3, 随机未交互物品2, 随机未交互物品1]`

这里用的是从序列末尾往前填充的方式，前面不足的位置用0填充（右对齐，左 padding），原因：Transformer模型中，位置编码是从左到右递增的，使用右对齐可以让模型更好地捕捉序列的时间顺序信息。

之后，`sample_function`的主要部分：

``` python
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    np.random.seed(SEED)
    uids = np.arange(1, usernum+1, dtype=np.int32)
    counter = 0
    while True:
        if counter % usernum == 0:
            np.random.shuffle(uids)
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample(uids[counter % usernum]))
            counter += 1
        result_queue.put(zip(*one_batch))
```

每个 `batch` 的数据 `one_batch` 包括 `batch_size` 个用户的数据，每个用户的数据是 `(uid, seq, pos, neg)`。

这里无限循环进行采样，每次采样 `batch_size` 个用户的数据，放入 `result_queue` 中，供训练时使用，按顺序从所有的`uids` 中取 `uid`，都取完一遍后重新打乱顺序。 `uids, seqs, poss, negs = zip(*one_batch)`。

- `uids`: `(batch_size, )` 用户ID列表
- `seqs`: `(batch_size, maxlen)` 用户历史交互物品序列
- `poss`: `(batch_size, maxlen)` 正例物品序列
- `negs`: `(batch_size, maxlen)` 负例物品序列

```
one_batch = [                   zip(                            (
  (uid1, seq1, pos1, neg1),       (uid1, seq1, pos1, neg1),       (uid1, uid2, uid3, ...),  
  (uid2, seq2, pos2, neg2), ==>   (uid2, seq2, pos2, neg2), ==>   (seq1, seq2, seq3, ...),
  (uid3, seq3, pos3, neg3),       (uid3, seq3, pos3, neg3),       (pos1, pos2, pos3, ...),
  ...                             ...                             (neg1, neg2, neg3, ...)
]                                 )                             )
```


最后的 `WarpSampler` 类：

``` python
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, 
                args=(User,
                      usernum,
                      itemnum,
                      batch_size,
                      maxlen,
                      self.result_queue,
                      np.random.randint(2e9)
                      )
                    )
                  )
            self.processors[-1].daemon = True # 随主进程关闭一起关闭
            self.processors[-1].start() # 子进程启动

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate() # 终止进程
            p.join() # 等待进程结束
```

这个实现比较抽象，多进程的过程是 `n_workers` 个 `sample_function` 进程在不断地往 `result_queue` 中放采样好的 `batch` 数据，主进程通过调用 `next_batch()` 方法从 `result_queue` 中获取一个 `batch` 数据。

不同的 `Process` 有不同的随机种子，保证采样的多样性，但是多个batch中可能会有重复的用户ID，而且对于相同的 `uid`，`seq` 和 `pos` 是相同的，只有 `neg` 是不同的，不太清楚这里的数据利用率。。。

但是最后采出来的一组数据就是上面的 `(uids, seqs, poss, negs)`。

## 模型

模型输入：`uids, seqs, poss, negs`

- `uids`: `(batch_size, )` 用户ID列表
- `seqs`: `(batch_size, maxlen)` 用户历史交互物品序列(在这里就是 `user_train[uids][-1-maxlen:-1]` 部分)，但是右对齐，左边padding成0
- `poss`: `(batch_size, maxlen)` 正例物品序列
- `negs`: `(batch_size, maxlen)` 负例物品序列

!!! warning 
    在这里取每个用户数据的时候没有使用sliding window: 直接取最后 `maxlen` 个交互物品作为 `seq`，对应的正例 `pos` 是 `seq` 后面的物品，所以 `uid` 相同的时候 `pos` 和 `seq` 也是相同的。

模型输出：`pos_logits` 和 `neg_logits`

- `pos_logits`: `(batch_size, maxlen)` 正例物品的预测得分
- `neg_logits`: `(batch_size, maxlen)` 负例物品的预测得分

![SASRec](images/generateive_recall.svg)

原文：

Create an item embedding matrix $M \in \mathbb{R}^{|\mathcal{I}| \times d}$: $d$ 是 latent dimensionality

初始化：

``` python
self.item_emb = torch.nn.Embedding(
    num_embeddings=item_num + 1, # 0 idx is used to padding
    embedding_dim=hidden_dim,
    padding_idx=0
)
```

因为在构造 `poss` 和 `seqs` 的时候，左边padding成0了，所以这里需要把 item embedding matrix中也设置 `padding_idx=0`，避免该位置参数变化。

Retrieve the input embedding matrix $E \in \mathbb{R}^{n\times d}$, where $E_i = M_{s_i}$

添加 Positional Embedding: Hence we inject a learnable position embedding $P \in \mathbb{R}^{n \times d}$ into the input embedding:

$$
\hat{E} = \begin{bmatrix} 
    M_{s_1} + P_1 \\
    M_{s_2} + P_2 \\
    \vdots \\
    M_{s_n} + P_n
\end{bmatrix} 
$$

Positional Embedding的初始化：

``` python
self.pos_emb = torch.nn.Embedding(
    num_embeddings=maxlen,
    embedding_dim=hidden_dim
)
```

III.C: We also apply a dropout layer on the embedding $\hat{E}$:

后面要接一个 emb_dropout:


