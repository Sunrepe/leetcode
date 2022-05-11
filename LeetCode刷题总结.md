# LeetCode刷题总结

## 绪论

### 1，解题思路

#### 1，复杂问题先排序

在二分、DP等多种类型问题里面，先进行数组排序，都会有利于求解！

例题：`p611, `

#### 2，双指针 (双向遍历)

除了简单的单向遍历以外，尽可能尝试形成两跟指针进行双向遍历，滑动窗口，有利于解决很多问题。

例题：`p209, p611`;

#### 3，倒序遍历

正向遍历思路比较容易想，但是很多时候采用倒序遍历可以降低复杂度！

例题：`p611, `

## 第一章：二分查找

### 二分模板

[主要参考](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/solution/tu-jie-er-fen-zui-qing-xi-yi-dong-de-jia-ddvc/)

#### 模板1

当我们将区间`[l, r]`划分成`[l, mid]`和`[mid+1, r]`时，其更新操作是`r = mid`或者`l = mid+1`，计算`mid`时不需要加`1`，即`mid = (l+r)/2`。

C++/java代码模板：

```c++
int bsearch_1(int l, int r)
{
    while (l < r)
    {
        int mid = (l + r)/2;
        if (check(mid)) r = mid;
        else l = mid + 1;
    }
    return r;
}
```

#### 模板2

当我们将区间`[l, r]`划分成`[l,mid-1]`和`[mid,r]`时，其更新操作是`r = mid-1`或者`l = mid`，为了防止死循环，计算`mid`时需要加`1`，即`mid = (l+r+1)/2`。

C++/java 代码模板：

```c++
int bsearch_2(int l, int r)
{
    while (l < r)
    {
        int mid = ( l + r + 1 ) /2;
        if (check(mid)) l = mid;
        else r = mid - 1;
    }
    return r;
}
```

**while循环结束条件是`l >= r`，但为什么二分结束时我们优先取`r`而不是`l`?**

二分的`while`循环的结束条件是`l >= r`，所以在循环结束时`l`有可能会大于`r`，此时就可能导致越界，因此，基本上二分问题优先取`r`都不会翻车。

#### Problem 209

> 给定一个含有 n 个正整数的数组和一个正整数 target 。
>
> 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
>
> 要求：分别实现O(n)和O($n\log n$)算法

##### 解决思路：滑动窗口遍历

对于O(n) 的算法，其实就是利用两个指针`pleft`、`pright`，对窗口进行滑动，先向右滑动`pright`直到满足要求，然后再滑动最左边的`pleft`直到不再满足要求。两个指针分别完成全部滑动即可找到。

```python
MAXN = 2e9
class eSolution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        l, total, res = 0, 0, MAXN
        for r in range(len(nums)):
            total += nums[r]
            while(l<=r and total>=target):
                res = min(res, r-l+1)
                total -= nums[l]
                l += 1
        return 0 if res == MAXN else res
```



#### problem 611

> 给定一个包含非负整数的数组 `nums` ，返回其中可以组成三角形三条边的三元组个数。
>
> `输入: nums = [2,2,3,4] ；输出: 3`
>
> *进阶：*
>
> 要求不计算重复的元素！

- 错误一：`Time_limit`：

  最基础的方法是暴力遍历，$O(n^3)$。

  $O(n^2 \log n)$ 的方法，思路是先排序，然后前两个数遍历，第三个数二分查找！但是仍然超出时间限制！

- 错误二：`0处理`，典型输入`[1,0,0,0,0,0]`

##### 解决思路：滑动窗口遍历

​	$O(n^2)$ 的方法：先排序；外层再**倒序遍历**（固定最大的数）；内层采用滑动窗口**双向遍历**找到区间两数之和大于最大数；具体而言，当遍历到某一区间符合要求时，

```python
class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        length = len(nums)
        nums.sort()
        res = 0
        for i in range(length-1, 1, -1):
            l,r = 0, i-1
            while(l<r):
                if nums[r]+nums[l] > nums[i]:
                    res += (r-l)
                    r -= 1
                else:
                    l += 1
        return res
```

#### problem 658

> 给定一个 **排序好** 的数组&nbsp;`arr` ，两个整数 `k` 和 `x` ，从数组中找到最靠近 `x`（两数之差最小）的 `k` 个数。返回的结果必须要是按升序排好的。
>
> 整数 `a` 比整数 `b` 更接近 `x` 需要满足：
>
> *   `|a - x| < |b - x|` 或者
> *   `|a - x| == |b - x|` 且 `a < b`
>
> **示例 ：**
>
> `输入：arr = [1,2,3,4,5], k = 4, x = 3；输出：[1,2,3,4]`
>
> **易错示例**：
>
> `输入：arr = [1,2,2,2,2,2,2,3,3], k = 3, x = 3；输出：[2,3,3]`

