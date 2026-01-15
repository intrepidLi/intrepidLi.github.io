---
title: Leetcode Problems
date: 2024-05-23
math: true
summary: Some Leetcode problems and solutions.

categories:
  - Algorithms
---



## Leetcode 42. Trapping Rain Water
链接：https://leetcode-cn.com/problems/trapping-rain-water


### 思路
每个地方能接的水量等于左右两边的最大高度中的最小值，减去当前高度。因此，可以用双指针法来解决这个问题。
1. 用两个指针`left`和`right`分别指向数组的两端，用`maxleft`和`maxright`分别记录左右两端的最大值。
2. 每次比较`maxleft`和`maxright`的大小，如果`maxleft`小于`maxright`，则计算`left`位置的水量，否则计算`right`位置的水量。
3. 每次计算完水量后，移动`left`或`right`指针。
4. 直到`left`和`right`指针相遇。
5. 时间复杂度为$O(n)$。
6. 空间复杂度为$O(1)$。

### 代码

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
        n = len(height)
        left, right = 0, n - 1
        maxleft, maxright = height[0], height[n - 1]
        ans = 0
        while left <= right:
            maxleft = max(height[left], maxleft)
            maxright = max(height[right], maxright)
            if maxleft < maxright:
                ans += maxleft - height[left]
                left += 1
            else:
                ans += maxright - height[right]
                right -= 1
        return ans
```

## Leetcode 151. Reverse Words in a String
链接：https://leetcode-cn.com/problems/reverse-words-in-a-string

### 思路
1. 先将字符串去除首尾空格，然后将字符串按空格分割。
2. 将分割后的字符串逆序拼接。
3. 时间复杂度为$O(n)$。
4. 空间复杂度为$O(n)$。
5. 注意：Python中字符串是不可变对象，因此不能直接修改字符串，需要先转换为列表。

### 代码

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(reversed(s.strip().split()))
```
