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

#### 搜索结果讨论

- **没找到**：在`r == 0`或`r == len-1` 时，需要单独讨论找到的结果的正确性。有时候可以合并！

#### 常见问题

- 找到`array`中最接近目标`x`的元素

  包含 绝对值最接近，或者` <= x` `< x` 的最大值，或者 `>= x` `> x`  的最小值。

  细节：比较mid 结果，更新搜索区间时，反向考虑！即，找`< x` 结果，需要去更新`>= x`的区间，

  > 方法：先手动生成一个接近目标值的序列，确定区间更新方式。生成区间时需要有重复元素！
  >
  > 距离：对于目标值`5`，则生成序列` [3,4,5,5,5,6,7]`, 并针对 `mid` 为该序列中的每一个数的时候如何进行更新！

  - 找到 `<x` 的最大值：最后一个小于x的数。

    ```python
    def search_less_x(nums,x):
        l, r = 0, len(nums)-1
        while(l<r):
            mid = l+r >> 1
            if nums[mid] < x:  # 条件设为小于x
                l = mid+1
            else:
                r = mid
        return r if （r==len(nums)-1 and nums[r]<x） else r-1
    ```

  - 找到 `<=x` 的最后一个数，条件设为 `nums[mid] <= x`， 返回`r-1`

### 练习题目

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
class Solution:
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

#### problem 300

> 给你一个整数数组 `nums` ，找到其中最长严格递增子序列的长度。
>
> **子序列&nbsp;**是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，`[3,6,2,7]` 是数组 `[0,3,1,6,2,2,7]` 的子序列。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**nums = [10,9,2,5,3,7,101,18]
> **输出：**4
> </pre>
>
> **提示：**
>
> *   1 &lt;= nums.length &lt;= 2500
> *   -10<sup>4</sup> &lt;= nums[i] &lt;= 10<sup>4</sup>
>
> **进阶：**
>
> *   你能将算法的时间复杂度降低到&nbsp;`O(n log(n))` 吗?

##### 解决思路：动态规划 

定义： `opt[i]`: 记录以`i `结尾的序列的 最长递增子序列长度，即数组`nums[0:i+1]`的最长子序列长度。

状态更新：
$$
\text{opt}[i]=\left\{
\begin{array}{l}
1,\quad & \text{nums}[i]< \text{num}[0:i]的每一项 \\
\max (\text{opt}[t]+1),& 对每一个\text{nums}[t]< \text{num}[i]的每一项
\end{array}  

\right.
$$
复杂度$O(n^2)$，一次遍历找到所有最优`opt[k]`；对于当前的`opt[k]`而言，遍历前面所有项，进行状态更新。

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        length = len(nums)
        opt = [1]
        # 一次遍历 O(n)
        for i in range(1, length):
            # 找到小于当前项的最大值
            maxx = 1
            for _ in range(i):
                if nums[_] < nums[i] and opt[_]+1 > maxx:
                    maxx = opt[_]+1
            opt.append(maxx)
        return max(opt)
```

##### 进阶思路：动态规划 + 二分

传统的动态规划方法复杂度较高的原因是：原数组是无序的，最优值opt定义也是无序的，在opt[k] 的更新过程中，必须重新遍历所有前面的k-1个数组，才能正确更新。

**复杂度：$O(n \log n)$**， 比纯dp快不少！

改进思路是改变最优值状态定义：

- 定义：`opt[i] `记录 长度为` i+1` 最优子序列末尾元素的最小值。

  说明：如果存在两个长度为`2`的最优子序列`[1,8]`和`[2,4] `则更新，`opt[2-1]=4`

- 状态更新
  $$
  \text{opt}=\left\{
  \begin{array}{l}
  \text{opt}[0]=\text{nums}[i],\quad & \text{nums}[i]< \text{opt}[0], \\
  \text{opt}[t+1]=nums[i],& 其中\text{opt}[t]是满足 < \text{num}[i]的最大值
  \end{array}  
  \right.
  $$

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        opt = [nums[0]]
        for i in range(1, len(nums)):
            l, r = 0, len(opt)-1
            # 二分查找 <x 的最大数
            while(l<r):
                mid = l+r >> 1
                if opt[mid]<nums[i]:
                    l = mid+1
                else:
                    r = mid
            # opt[r-1] 是目标结果，opt[r]是第一个不满足要求的数，刚好需要调整。
            if r==len(opt)-1 and opt[r]<nums[i]:  # 末尾特殊情况，此时满足<x 要求
                opt.append(nums[i])
            else:
                opt[r] = min(opt[r], nums[i])
        return len(opt)
```

​	注意，在上述实现中，`if`部分表示 ：找到了目标，且目标是`opt`数组最后一个元素，此时可以形成更长的递增子序列。

​	**再进阶**：如果要求最长非降子序列长度，只需要第8行 判断 ` if opt[mid] <= nums[i]` 即可

#### problem 1760

> 给你一个整数数组&nbsp;`nums`&nbsp;，其中&nbsp;`nums[i]`&nbsp;表示第&nbsp;`i`&nbsp;个袋子里球的数目。同时给你一个整数&nbsp;`maxOperations`&nbsp;。
>
> 你可以进行如下操作至多&nbsp;`maxOperations`&nbsp;次：
>
> *   选择任意一个袋子，并将袋子里的球分到&nbsp;2 个新的袋子中，每个袋子里都有 **正整数**&nbsp;个球。
>
>     * 比方说，一个袋子里有&nbsp;`5`&nbsp;个球，你可以把它们分到两个新袋子里，分别有 `1`&nbsp;个和 `4`&nbsp;个球，或者分别有 `2`&nbsp;个和 `3`&nbsp;个球。
>
> 你的开销是单个袋子里球数目的 **最大值**&nbsp;，你想要 **最小化**&nbsp;开销。
>
> 请你返回进行上述操作后的最小开销。
>
> 
>
> **示例 1：**
>
> <pre>**输入：**nums = [2,4,8,2], maxOperations = 4
> **输出：**2
> **解释：**
> - 将装有 8 个球的袋子分成装有 4 个和 4 个球的袋子。[2,4,**8**,2] -&gt; [2,4,4,4,2] 。
> - 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,**4**,4,4,2] -&gt; [2,2,2,4,4,2] 。
> - 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,**4**,4,2] -&gt; [2,2,2,2,2,4,2] 。
> - 将装有 4 个球的袋子分成装有 2 个和 2 个球的袋子。[2,2,2,2,2,**4**,2] -&gt; [2,2,2,2,2,2,2,2] 。
> 装有最多球的袋子里装有 2 个球，所以开销为 2 并返回 2 。
> </pre>
>
> **示例 3：**
>
> <pre>**输入：**nums = [7,17], maxOperations = 2
> **输出：**7
> </pre>
>
> &nbsp;
>
> **提示：**
>
> *   1 &lt;= nums.length &lt;= 10<sup>5</sup>
> *   1 &lt;= maxOperations, nums[i] &lt;= 10<sup>9</sup>