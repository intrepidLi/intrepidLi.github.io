---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "RecDataset"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Summary", "Dataset"]
categories: ["Summary", "Dataset"]
date: 2025-12-24T16:29:10+08:00
lastmod: 2025-12-24T16:29:10+08:00
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

## Movie-Lens 

ml-1m 数据集：

https://files.grouplens.org/datasets/movielens/ml-1m.zip

### `movies.dat` 

`MovieID::Title::Genres`

- Titles are identical to titles provided by the IMDB (including
year of release)
- Genres are pipe-separated and are selected from the following genres:
    - Action
    - Adventure
    - Animation
    - Children's
    - Comedy
    - Crime
    - Documentary
    - Drama
    - Fantasy
    - Film-Noir
    - Horror
    - Musical
    - Mystery
    - Romance
    - Sci-Fi
    - Thriller
    - War
    - Western
- Some MovieIDs do not correspond to a movie due to accidental duplicate entries and/or test entries
- Movies are mostly entered by hand, so errors and inconsistencies may exist

例子：

```latex
1::Toy Story (1995)::Animation|Children's|Comedy
2::Jumanji (1995)::Adventure|Children's|Fantasy
3::Grumpier Old Men (1995)::Comedy|Romance
4::Waiting to Exhale (1995)::Comedy|Drama
5::Father of the Bride Part II (1995)::Comedy
```

### `ratings.dat` 

`UserID::MovieID::Rating::Timestamp`

类似：

```latex
1::1193::5::978300760
1::661::3::978302109
1::914::3::978301968
1::3408::4::978300275
```

- UserIDs range between 1 and 6040
- MovieIDs range between 1 and 3952
- Ratings are made on a 5-star scale (whole-star ratings only)
- Timestamp is represented in seconds since the epoch as returned by time(2)
- Each user has at least 20 ratings

### `users.dat`  

`UserID::Gender::Age::Occupation::Zip-code`

- Gender is denoted by a "M" for male and "F" for female
- Age is chosen from the following ranges:
    - 1: "Under 18"
    - 18: "18-24"
    - 25: "25-34"
    - 35: "35-44"
    - 45: "45-49"
    - 50: "50-55"
    - 56: "56+"
- Occupation is chosen from the following choices:
    - 0: "other" or not specified
    - 1: "academic/educator"
    - 2: "artist"
    - 3: "clerical/admin"
    - 4: "college/grad student"
    - 5: "customer service"
    - 6: "doctor/health care"
    - 7: "executive/managerial"
    - 8: "farmer"
    - 9: "homemaker"
    - 10: "K-12 student"
    - 11: "lawyer"
    - 12: "programmer"
    - 13: "retired"
    - 14: "sales/marketing"
    - 15: "scientist"
    - 16: "self-employed"
    - 17: "technician/engineer"
    - 18: "tradesman/craftsman"
    - 19: "unemployed"
    - 20: "writer"

Examples:

```latex
1::F::1::10::48067
2::M::56::16::70072
3::M::25::15::55117
4::M::45::7::02460
5::M::25::20::55455
```

在SASRec中用的 `ml-1m.txt` 处理方法：

实际只处理 `ratings.dat`，具体处理过程：

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

训练脚本：

``` bash
python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

训练 epoch 代码：

``` python
for epoch in range(1, args.num_epochs + 1):

        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            u, seq, pos, neg = sampler.next_batch()
            auc, loss, _ = sess.run([model.auc, model.loss, model.train_op],
                                    {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg,
                                     model.is_training: True})

        if epoch % 20 == 0:
            t1 = time.time() - t0
            T += t1
            print 'Evaluating',
            t_test = evaluate(model, dataset, args, sess)
            t_valid = evaluate_valid(model, dataset, args, sess)
            print ''
            print 'epoch:%d, time: %f(s), valid (NDCG@10: %.4f, HR@10: %.4f), test (NDCG@10: %.4f, HR@10: %.4f)' % (
            epoch, T, t_valid[0], t_valid[1], t_test[0], t_test[1])

            f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            f.flush()
            t0 = time.time()

```




