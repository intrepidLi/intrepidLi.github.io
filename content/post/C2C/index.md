---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "C2C"
subtitle: ""
summary: ""
authors: []
tags: []
categories: []
date: 2026-01-07T10:16:43+08:00
lastmod: 2026-01-07T10:16:43+08:00
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

代码：

加载 两个模型:

`base_model: Qwen/Qwen3-0.6B`, `teacher_model: Qwen/Qwen2.5-0.5B-Instruct`

都是 `torch.bfloat16` 精度加载到 `eval()` 模式

加载 'projectors':

``` python
# Load projectors
num_projectors = len([f for f in os.listdir(checkpoint_dir) if re.match(r"projector_\d+\.pt", f)])
projector_list = []
for t in range(num_projectors):
    json_cfg = os.path.join(checkpoint_dir, f"projector_{t}.json")
    proj = load_projector(json_cfg)
    proj = proj.to(device)
    pt_path = os.path.join(checkpoint_dir, f"projector_{t}.pt")
    if os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location=device)
        proj.load_state_dict(state_dict, strict=False)
    projector_list.append(proj)
```

总共的 `num_projectors=28`, 剩下的都是加载 `Testing_proj/C2C/C2C_Fuser/qwen3_0.6b+qwen2.5_0.5b_Fuser` 里面所有的 `projector_x.json` 和 `projector_x.pt`

`projector_list[0]` 的架构：

```
C2CProjector(
  (key_in): Linear(in_features=1152, out_features=1024, bias=True)
  (value_in): Linear(in_features=1152, out_features=1024, bias=True)
  (key_mlp1): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (value_mlp1): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (key_scalar_mlp2): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (value_scalar_mlp2): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (key_scalar_head): Linear(in_features=1024, out_features=8, bias=True)
  (value_scalar_head): Linear(in_features=1024, out_features=8, bias=True)
  (key_proj_mlp2): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (value_proj_mlp2): RegularMLP(
    (blocks): ModuleList(
      (0): StandardFFNLayer(
        (norm): RMSNorm((1024,), eps=1e-06, elementwise_affine=True)
        (w1): Linear(in_features=1024, out_features=1024, bias=False)
        (w2): Linear(in_features=1024, out_features=1024, bias=False)
        (drop): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
    )
  )
  (key_proj_out): Linear(in_features=1024, out_features=1024, bias=True)
  (value_proj_out): Linear(in_features=1024, out_features=1024, bias=True)
)
```

`base_model: Qwen/Qwen3-0.6B`, `teacher_model: Qwen/Qwen2.5-0.5B-Instruct`, `base_model_idx=0`，就是 `Qwen3-0.6B` 对应 `self.model_list[0]`

在 `inference_example.py` 中的 `load_rosetta_model` 函数，返回的 `rosetta_mode` 和 `tokenizer` 分别是 一个 `RosettaModel` 实例 和 `base_model` 的 `tokenizer`，在这里就是 `Qwen3-0.6B` 的 `tokenizer`  

``` python
prompt = [{"role": "user", "content": "Say hello in one short sentence."}]
input_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=False)
inputs = tokenizer(input_text, return_tensors="pt").to(device)
```

`input_text` 打印出来：

`<|im_start|>user\nSay hello in one short sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`

`inputs` 打印出来：

`{'input_ids': tensor([[151644,    872,    198,  45764,  23811,    304,    825,   2805,  11652,
             13, 151645,    198, 151644,  77091,    198, 151667,    271, 151668,
            271]], device='cuda:0'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
       device='cuda:0')}`

`RosettaModel` 的 `generate` 函数：

`input_ids: torch.Size([1, 19])`，不是一个 list，所以 `base_input_ids_for_len = input_ids`

`gen_cfg` 和 `cfg_obj`:

``` bash
GenerationConfig {
  "bos_token_id": 151643,
  "do_sample": true,
  "eos_token_id": [
    151645,
    151643
  ],
  "pad_token_id": 151643,
  "temperature": 0.6,
  "top_k": 20,
  "top_p": 0.95
}
```

现在进行 `forward`:

``` python
prefill_output = self.forward(
    kv_cache_index=kv_cache_index,
    input_ids=input_ids,
    attention_mask=attention_mask,
    position_ids=position_ids,
    past_key_values=past_key_values,
    use_cache=use_cache,
    *args,
    **kwargs,
)
```

`test.py` 中设置了一个 `instruction_index`:

``` python
instruction_index = torch.tensor([1, 0], dtype=torch.long).repeat(inputs['input_ids'].shape[1] - 1, 1).unsqueeze(0).to(device)
```

`repeat` 进行 内存复制，并且 `repeat` 的参数维度 $\ge $ 原始 tensor 维度，像上面这个情况 `[1,0]` 维度是1，但是 `repeat` 的参数有2个，会先给 `[1,0]` 隐式补一个维度变成 `[2] -> [1,2]`，然后再进行 `repeat`，所以 `instruction_index` 最终 shape 是 `[1,18,2]`(`(1, L -1, 2)`), `L` 是 `input_ids` 的长度 19。

一个 `label_index`:

``` python
label_index = torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(device)
```

dimension是 `[1, 1, 2]`

`kv_cache_index = [instruction_index, label_index]`: `[tensor of shape (1,18,2), tensor of shape (1,1,2)]`

`section_lengths` 记录 `kv_cache_index` 里面每个 tensor 的第二维长度：`[18, 1]`

`base_input_ids` 的 shape是 `[1,19]`(`[B, L]`)

先 Qwen3-0.6B forward一下：

``` python
output = self.model_list[self.base_model_idx].forward(
    input_ids=prefill_input_ids,
    attention_mask=prefill_attention_mask, 
    position_ids=prefill_position_ids,
    past_key_values=curr_base_kv_cache,
    labels=prefill_labels,
    use_cache=use_cache, 
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    *args,
    **kwargs
)
```

这个 `output` 是个 `CausalLMOutputWithPast` 类，里面含有两个键 `odict_keys(['logits', 'past_key_values'])`

`base_input_ids: tensor([[151644,    872,    198,  45764,  23811,    304,    825,   2805,  11652, 13, 151645,    198, 151644,  77091,    198, 151667, 271, 151668, 271]], device='cuda:0')`

第一轮：

`prefill_input_ids: tensor([[151644,    872,    198,  45764,  23811,    304,    825,   2805,  11652, 13, 151645,    198, 151644,  77091,    198, 151667,    271, 151668]], device='cuda:0')`


















