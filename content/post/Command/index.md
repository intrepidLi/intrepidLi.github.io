---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "Command"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2026-01-05T15:58:57+08:00
lastmod: 2026-01-05T15:58:57+08:00
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

## Common Commands

Conda create a new environment to specific python version and path:

```bash
conda create -p ./envs/python38 python=3.8 -y
```

`-p` or `--prefix`: specify the path to the environment location.

`-y` : automatically confirm the installation without prompting.(`yes`)

如果前面不会显示 `(base)`，可以通过下面命令激活 base 环境：

``` bash
cd /opt/conda/bin
source activate root
```

在显示`(base)`之后激活某个环境：

``` bash
source/conda activate XXX  # XXX is the environment name or path
```

查看 `conda` 的位置：

``` bash
which conda

type __conda_exe

echo $CONDA_PREFIX
```

huggingface download to local dir:

``` bash
hf download edinburgh-dawg/mmlu-redux --local-dir /home/mli6/datastore/yuliang_use/dataset/mmlu-redux --repo-type model/dataset

hf download Qwen/Qwen3-0.6B --local-dir /home/mli6/datastore/yuliang_use/models/Qwen3-0.6B --repo-type model

hf download Qwen/Qwen3-1.7B --local-dir /home/mli6/datastore/yuliang_use/models/Qwen3-1.7B --repo-type model

hf download gradientai/Llama-3-8B-Instruct-262k --local-dir /home/mli6/datastore/yuliang_use/models/Llama-3-8B-Instruct-262k --repo-type model


local_dir: /home/mli6/datastore/yuliang_use/dataset/mmlu-redux
```

``` bash
srun --partition=ICF-Research --gres=gpu:1 --cpus-per-task=8 --ntasks=1 --pty bash
```

### pdb 调试

在代码中插入：

``` python
import pdb; pdb.set_trace()
```

或者

``` bash
python -m pdb your_script.py
```

然后运行脚本，程序会在 `set_trace()` 处暂停，进入交互式调试模式。

在调试模式下，可以使用以下命令：

- `n` (next): 执行下一行代码
- `c` (continue): 继续执行程序直到下一个断点
- `q` (quit): 退出调试器
- `s` (step): 进入函数调用
- `l` (list): 显示当前代码行的上下文
- `p variable_name`: 打印变量的值
- `b filename:line_number`: 在指定文件的指定行设置断点
  - 注意像debug 类似 `model.generate` 的函数的时候就用这个方法

注意设置断点的时候尽量不要打在函数定义 `def` 的位置，而是打在函数体内的某一行，否则可能无法命中断点。

`b /home/mli6/datastore/yuliang_use/projects/Testing_proj/C2C/rosetta/model/wrapper.py:428`

`b /home/mli6/datastore/yuliang_use/projects/Testing_proj/C2C/rosetta/model/wrapper.py:702`

