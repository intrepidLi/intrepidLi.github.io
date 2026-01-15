---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "VLM"
subtitle: ""
summary: ""
authors: [admin]
tags: ["VLM", "Summary"]
categories: ["VLM", "Summary"]
date: 2025-12-25T15:32:28+08:00
lastmod: 2025-12-25T15:32:28+08:00
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

使用 `transformers` 调用 VLM 进行推理：

`InternVL3`:

``` python
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

torch_device = "cuda"
model_checkpoint = "OpenGVLab/InternVL3-1B-hf"
processor = AutoProcessor.from_pretrained(model_checkpoint)
model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, device_map=torch_device, torch_dtype=torch.bfloat16)

messages = [
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://llava-vl.github.io/static/images/view.jpg"},
                {"type": "text", "text": "Write a haiku for this image"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"},
                {"type": "image", "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg"},
                {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
            ],
        },
    ],
]

inputs = processor.apply_chat_template(messages, padding=True, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device).to(torch.bfloat16)

output = model.generate(**inputs, max_new_tokens=25)

decoded_outputs = processor.batch_decode(output, skip_special_tokens=True)
decoded_outputs
```
