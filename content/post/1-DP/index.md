---
# Documentation: https://hugoblox.com/docs/managing-content/

title: "1 Dimension DP"
subtitle: ""
summary: ""
authors: [admin]
tags: ["Algorithms"]
categories: ["Algorithms"]
date: 2025-12-23T17:14:45+08:00
lastmod: 2025-12-23T17:14:45+08:00
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

## Leetcode 139 Word Break

Given a string `s` and a dictionary of strings `wordDict`, return `true` if `s` can be segmented into a space-separated sequence of one or more dictionary words.

**Note** that the same word in the dictionary may be reused multiple times in the segmentation.

**Example 1:**

> Input: `s = "leetcode", wordDict = ["leet","code"]`
> Output: `true`
> Explanation: Return `true` because "leetcode" can be segmented as "leet code".

这个题是 1维DP，思路：用 `dp[i]` 表示 `s[0:i]` 能否被 wordDict 拆分，最终结果就是 `dp[n]`。

输入是一个 字符串（序列），所以切分子问题直接用 prefix。

- Sub-problems: prefix `dp[i]` indicates whether `s[0:i]` can be segmented
- Relate: `dp[i]` = any(`dp[j]` 可以被切分 and `s[j:i]` in `wordDict` for `j` in `[0, i)`)
- Topo order: increasing `i` from `0` to `n`, increasing `j` from `0` to `i`
- Base cases: `dp[0] = True` (empty string can be segmented)
- Original problem: `dp[n]`
- Time: `O(n^2)`, Space: `O(n)`

```python
class Solution(object):
    def wordBreak(self, s, wordDict):
        word_set = set(wordDict) # 转换为集合，提升查找速度
        n = len(s)
        # dp[i] 表示 s 的前 i 个字符是否可以拆分
        dp = [False] * (n + 1)
        dp[0] = True # 初始状态
        
        for i in range(1, n + 1):
            for j in range(i):
                # 如果 s[0:j] 合法 且 s[j:i] 在字典中
                if dp[j] and s[j:i] in word_set:
                    dp[i] = True
                    break # 只要找到一种方案，就可以停止当前 i 的搜索
                    
        return dp[n]
```

