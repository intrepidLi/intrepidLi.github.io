---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "LongContext"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2026-01-07T13:06:29+08:00
lastmod: 2026-01-07T13:06:29+08:00
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

这是 `PersonaMem` Benchmark 的数据：



`questions_32k.csv`:

```
persona_id,question_id,question_type,topic,context_length_in_tokens,context_length_in_letters,distance_to_ref_in_blocks,distance_to_ref_i
n_tokens,num_irrelevant_tokens,distance_to_ref_proportion_in_context,user_question_or_message,correct_answer,all_options,shared_context_i
d,end_index_in_shared_context
```

解析 `json/jsonl` 的脚本：`parse_json.py`:

``` python
import json
import os
from collections import defaultdict

def get_value_type(value):
    """返回值的类型名称，如果是列表则标注为 list"""
    if value is None:
        return "None"
    return type(value).__name__

def extract_structure(data, prefix='', structure=None):
    """
    递归提取 JSON 结构的键和类型
    """
    if structure is None:
        structure = defaultdict(set)

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                extract_structure(value, new_key, structure)
            elif isinstance(value, list):
                structure[new_key].add("list")
                # 如果列表不为空，递归检查第一个元素以了解元素类型
                if len(value) > 0:
                    extract_structure(value[0], f"{new_key}[]", structure)
            else:
                structure[new_key].add(get_value_type(value))
    
    elif isinstance(data, list):
        # 处理顶层就是列表的情况
        for item in data:
            extract_structure(item, prefix, structure)
            
    return structure

def analyze_file(file_path):
    """分析 json 或 jsonl 文件"""
    all_structure = defaultdict(set)
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return

    print(f"正在分析文件: {file_path} ...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 判断文件扩展名
            if file_path.endswith('.jsonl'):
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        data = json.loads(line)
                        extract_structure(data, structure=all_structure)
            else:
                # 普通 json 文件
                data = json.load(f)
                extract_structure(data, structure=all_structure)

        # 打印结果
        print("\n" + "="*50)
        print(f"{'Key 名称':<40} | {'Value 类型'}")
        print("-"*50)
        
        # 排序后输出
        for key in sorted(all_structure.keys()):
            types = ", ".join(all_structure[key])
            print(f"{key:<40} | {types}")
        print("="*50)

    except json.JSONDecodeError as e:
        print(f"解析失败，JSON 格式错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 在这里输入你的文件名
    target_file = "shared_contexts_32k.jsonl"  # 或者 "data.json"
    analyze_file(target_file)
```

最后 `shared_contexts_32k.jsonl` 得到的结果：

所有 `key` 都是 `shared_context_id`， `value` 是个 `list`，这个 `list` 里面是多轮对话，每轮对话是个 `dict`，包含 `role` 和 `content` 两个 `str` 字段。

`role` 有两类：`system` 和 `user`。