---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "LatentMAS"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Papers", "latent_memory"]
categories: ["Papers", "latent_memory"]
date: 2025-12-24T16:25:54+08:00
lastmod: 2025-12-24T16:25:54+08:00
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

# LatentMAS-2511.20639

## 解决问题

Multi Agents 能否不依赖 自然语言文本，实现纯 Latent Space的协作。


## 方法

- 潜思维生成：不解码 token，而是自回归地直接用最后一层隐藏状态 $h_t$ 作为下一步输入

- 每个智能体完成 $m$ 步潜推理后，一次性抽取全部层级的 KV-cache
- 用KV-cache对齐和拼接保证信息不丢失

## 代码

### 数据

`./data/medqa.json`: 

``` json
[
  {
    "idx": 0,
    "question": "A 34-year-old man comes to the physician because of a 3-week history of colicky abdominal pain and diarrhea. He has bowel movements 10–12 times daily; the st
ool contains blood and mucus. He constantly has the urge to defecate. His vital signs are within normal limits. Examination of the abdomen shows diffuse tenderness to palpati
on. Serum concentration of C-reactive protein is 20 mg/L (N<10). Colonoscopy shows a bleeding, ulcerated rectal mucosa with several pseudopolyps. Which of the following is th
is patient at greatest risk of developing?\nA. Hemolytic uremic syndrome\nB. Oral ulcers\nC. Colorectal cancer\nD. Pancreatic cancer\n\nA. Hemolytic uremic syndrome\nB. Oral 
ulcers\nC. Colorectal cancer\nD. Pancreatic cancer",
    "options": [
      "A. Hemolytic uremic syndrome",
      "B. Oral ulcers",
      "C. Colorectal cancer",
      "D. Pancreatic cancer"
    ],
    "answer": "Colorectal cancer",
    "gen_text_store": "",
    "pid": "0",
    "query": "A 34-year-old man comes to the physician because of a 3-week history of colicky abdominal pain and diarrhea. He has bowel movements 10–12 times daily; the stool
 contains blood and mucus. He constantly has the urge to defecate. His vital signs are within normal limits. Examination of the abdomen shows diffuse tenderness to palpation.
 Serum concentration of C-reactive protein is 20 mg/L (N<10). Colonoscopy shows a bleeding, ulcerated rectal mucosa with several pseudopolyps. Which of the following is this 
patient at greatest risk of developing?\nA. Hemolytic uremic syndrome\nB. Oral ulcers\nC. Colorectal cancer\nD. Pancreatic cancer\nChoose the correct option.",
    "image": null
  },
  ...
]
```

每个 item 的数据样式：

``` json
{
  "idx": ...,
  "question": "...",
  "options": ["...", "...", "...", "..." ],
  "answer": "...",
  "gen_text_store": "",
  "pid": "...",
  "query": "...",
  "image": null
}
```

这个 `question` 很怪，重复了两遍选项，也没说 `Choose the correct option`，代码里用的是 `query` 字段： `题干 + 选项 + Choose the correct option`。

`data.py` 中：

``` python
def load_medqa(split=None, subset=None, cache_dir=None):

    ds = load_dataset("json", data_files="./data/medqa.json", split='train')
    for item in ds:
        question = item["query"]
        raw_answer = str(item["answer"])

        choice_map = {"0":"A", "1":"B", "2":"C", "3":"D"}

        for idx, op in enumerate(item['options']):
            if raw_answer in op:
                answer = choice_map[str(idx)].lower()
                break

        gold = normalize_answer(answer)

        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }

def normalize_answer(ans: Optional[str]) -> Optional[str]:
    if ans is None:
        return None
    return ans.strip().lower()
```

这段代码最后会输出一个 `generator`(`yield`):

``` json
{
  "question": 原来的 query 字段内容,
  "solution": "a/b/c/d",
  "gold": "a/b/c/d"
}
```

这个 `gold` 和 `solution` 完全一样，不太懂

`args.max_samples=-1`: 用全部数据

`medqa.json` 总共300条数据

`args.generate_bs=20`: 

``` python
for item in dataset_iter:
    if processed >= args.max_samples:
        break
    batch.append(item)
    if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
        processed, preds = process_batch(
            method,
            batch,
            processed,
            preds,
            progress,
            args.max_samples=300,
            args,
        )
        batch = []
        if processed >= args.max_samples:
            break
```

上面这段：有20条数据就 `process_batch`，

``` python
# Main processing function for each batch
def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    if args.method == "latent_mas" and args.use_vllm: 
        results = method.run_batch_vllm(current_batch) 
    else:
        results = method.run_batch(current_batch)
    if len(results) > remaining:
        results = results[:remaining]
    batch_start = processed
    for offset, res in enumerate(results):
        preds.append(res)
        problem_idx = batch_start + offset + 1
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds
```

`processed` 记录已经处理了多少数据，`preds` 记录所有结果，感觉 `process_batch` 函数中的 `:remaining` 没啥用，因为上面已经控制了 `batch` 的大小。

现在就是 传入 `method.run_batch(current_batch)` 进行推理, 每一组的数据是 20 条：

``` json
[
  {
    "question": "...",
    "solution": "a/b/c/d",
    "gold": "a/b/c/d"
  }, 
  ...
]
```

`BaselineMethod` 类:

``` python
class BaselineMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens # 2048
        self.temperature = temperature # 0.7
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = args.task
    
    def run_batch(self, items: List[Dict]) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        batch_messages = [
            build_agent_messages_single_agent(question=item["question"], args=self.args)
            for item in items
        ]
        prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
            batch_messages, add_generation_prompt=True
        )
        
        if self.use_vllm:
            generated_batch = self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            generated_batch, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )

        results: List[Dict] = []
        
        for idx, item in enumerate(items):
            generated_text = generated_batch[idx]
            
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(generated_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(generated_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            else:
                pred = normalize_answer(extract_gsm8k_answer(generated_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].to("cpu").tolist()
            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
            }
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": generated_text,
                    "agents": [agent_trace],
                    "correct": ok,
                }
            )
        return results
```

这个 `run_batch` 函数

`prompts.py` 中，

`system_message`: `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`

现在的写法是只能做 Qwen 模型：

`user_content`:

- `gsm8k, aime2024, aime2025`: 数学推理
  - ``` txt
    Target Question: {question}
    You are a helpful assistant.
    You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
    Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
    ```
- `arc_easy, arc_challenge, gpqa, medqa`: 多项选择
  - ``` txt
    Target Question: {question}
    You are a helpful assistant.
    You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
    Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.
    Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
    ```
- `mbppplus, humanevalplus`: 代码补全
  - ``` txt
    Target Question: {question}

    You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
    ```python
    import math
    def add(a, b):
        return a + b
    ```
    Do not add any other contents inside the markdown code block.
    Now, reason step by step and output the final answer:
    ```
- `winogrande`: 双选题
  - ``` txt
    Target Question: {question}

    You are a helpful assistant.

    You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
    Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

    Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
    ``` 
- 其他任务:
  - ``` txt
    Question: {question}

    You are a helpful assistant.

    You must reason step-by-step to solve the question without outputting other irrelevant information.
    Present your reasoning, and then clearly state your final answer at the end.
    ```

`return` 的结果：

``` python
return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]
```




