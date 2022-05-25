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

## 第一章：基础数据结构

### 1，set - 集合

#### problem 1

> 给定一个整数数组 `nums`&nbsp;和一个整数目标值 `target`，请你在该数组中找出 **和为目标值 **_`target`_&nbsp; 的那&nbsp;**两个**&nbsp;整数，并返回它们的数组下标。
>
> 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
>
> 你可以按任意顺序返回答案。
>
> &nbsp;
>
> **示例 1：**
>
> > **输入：**nums = [2,7,11,15], target = 9
> > **输出：**[0,1]
> > **解释：**因为 nums[0] + nums[1] == 9 ，返回 [0, 1] 。
>
> **易错示例 2：**
>
> > **输入：**nums = [3, 3], target = 6
> > **输出：**[0,1]
>
> **提示：**
>
> *   2 &lt;= nums.length &lt;= 10<sup>4</sup>
> *   -10<sup>9</sup> &lt;= nums[i] &lt;= 10<sup>9</sup>
> *   -10<sup>9</sup> &lt;= target &lt;= 10<sup>9</sup>
> *   **只会存在一个有效答案**

**思路分析：**

- `O(N)`的方法，只要使用hash方法，枚举`num`时，查找是否存在`hash(target-num)`即可
- *细节*：首先创建空哈希表，对于每一个 `x`，我们首先查询哈希表中是否存在 `target - x`，然后将 `x` 插入到哈希表中，即可保证不会让 `x` 和自己匹配。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        cmp = set()
        for i in range(len(nums)):
            if target-nums[i] in cmp:
                return [nums.index(target-nums[i]), i]
            cmp.add(nums[i])
```

#### problem 217



### 2，线段树

### 3，cluster - 并查集 

[参考来源：花花酱](https://www.youtube.com/watch?v=VJnUwsE4fWA&ab_channel=HuaHua)

在图论中，连通块可以用`并查集`来维护。

- 核心算法

  - `Find(x)`: 找到`root/cluster-id`, 不断沿着路径寻根
  - `Union(x, y)`： 将两个`cluster` 合并

- 复杂度

  - `Find(x)`：$O(\alpha(N))^*\approx O(1)  $
  - `Union(x, y)`：$O(\alpha(N))^*\approx O(N)  $
  - 空间复杂度：$O(N)$

- 两个核心优化技巧

  - **路径压缩**：采用递归实现

    第一次搜索的时候，将路径上所有节点的父节点全部改成`root`

    <img src=".\images\image-20220516210002292.png" alt="image-20220516210002292" style="zoom: 50%;" />

  - **Union by Rank**：

    `Rank`一般指混乱度，含义接近于树的高度。把rank低的合并到rank高的树上。

    ![](.\images\Snipaste_2022-05-16_21-04-20.png)

  - 伪代码

    

**problem 1020**

本题更简单快速的做法是直接`bfs`,详见图论中该题解法。

此处列出大致伪代码

```python
class UnionFind:
    def __init__(self, grid: List[List[int]]):
        m, n = len(grid), len(grid[0])
        self.parent = [0] * (m * n)  # 有几个节点就要形成几个长度，告知并查集的可能长度
        self.rank = [0] * (m * n)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.onEdge[x] |= self.onEdge[y]
        elif self.rank[x] < self.rank[y]:
            self.parent[x] = y
            self.onEdge[y] |= self.onEdge[x]
        else:
            self.parent[y] = x
            self.onEdge[x] |= self.onEdge[y]
            self.rank[x] += 1
```





## 第二章：二分查找

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

#### 模板3：红蓝划分 **good**

[参考：来自B站五点七边](https://www.bilibili.com/video/BV1d54y1q7k7)

将原数组划分为红蓝两个区域，并根据要求计算是否属于蓝区。

几个细节：

- 初始化为-1，和N，这样可以明确区分是否找到
- 循环结束条件是`l+1 == r`，可以确保找到了边界，但是对于返回结果是 `l or r`需要再思考

实现：

```python
l, r = -1, N
while l+1 != r:
    mid = l+r >> 1
    if IsBlue(mid, x):
        l = mid
    else:
        r = mid
```

具体细节：

![](.\images\二分红蓝.png)

上图中的边界区分非常清晰！

- 红区，在x右侧，但是要根据是否含有等于号区分细节边界
- 蓝区，在x左侧； 

*说明*：该方法简单易懂，而且模型明确。在复习时可以快速上手！模板1和模板2相对来说也比较容易入手，但是细节还需要继续背一下！

**使用方法：**



#### 搜索结果讨论

- **没找到**：在`r == 0`或`r == len-1` 时，需要单独讨论找到的结果的正确性。有时候可以合并！

#### 常见问题

##### 问题1： 4类基本查找 

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

##### 问题2：最接近目标值`x`

- 在已排序数组中找到最接近目标值`x`的数。

- 红蓝划分模板。思路和找等于`x`的数一样：如果数组中没有和`！x`相等的数，找到的一定是最接近的数。

  但是此时，需要讨论分别`l、r`的最优性，以及要区分是否找到！

  ```python
  l, r = -1, n
  while l + 1 < r:
      mid = l + r >> 1
      if ns[mid] < nums2[i]:
          l = mid
      else:
          r = mid
  res = None
  if l == -1:
      res = r
  elif r == n:
      res = l
  elif abs(nums2[i] - ns[l]) <= abs(nums2[i] - ns[r]):
      res = l
  else:
      res = r
  # res 为所求结果的下标。在实际应用中可以简化这个过程。
  ```

  

##### 问题3：最大值最小化

- `>=x`模板

- 原始：分割数组最大值最小化；给定一个非负整数数组 `nums` 和一个整数&nbsp;`m` ，你需要将这个数组分成&nbsp;`m`_&nbsp;_个非空的连续子数组。设计一个算法使得这&nbsp;`m`_&nbsp;_个子数组各自和的最大值最小。详见 `p410`


- 进阶：给定一个数组，将其划分成 `M` 份，使得每份元素之和最大值最小，每份可以任意减去其中一个元素。详见 `LCP 12`
- 进阶：将数组里的数切割，使得数组大小变为`N + M`份；详见 `p1760`

##### 问题4：最小值最大化 

- `<=x`模板

- 原始：将一个数组`array`分成`m`段，使得每段之间的距离的最小值最大化。详见`p1552`

##### 问题5：二维搜索

- 原始数组是二维数组，在每行每列上都保持有序关系！
- 

### 练习题目

#### Problem 209

> 给定一个含有 n 个正整数的数组和一个正整数 target 。
>
> 找出该数组中满足其和 ≥ target 的长度最小的 连续子数组 [numsl, numsl+1, ..., numsr-1, numsr] ，并返回其长度。如果不存在符合条件的子数组，返回 0 。
>
> 要求：分别实现O(n)和O($n\log n$)算法

**解决思路：滑动窗口遍历**

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

**解决思路：滑动窗口遍历**

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

#### problem 658  **try**

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

**思路1**：较复杂

- 找到最接近x的数：对于排序好的数组，遍历一次是O(n)的方法，但是还有更简单的的直接`o(log n)`的二分查找。

- `k`个数定位：根据找到的最优`x`，向两次扩散`k`个数即可

  ```python
  class Solution:
      def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
          length = len(arr)
  
          # 二分找到最接近x的数
          l, r = 0, len(arr) - 1
          while (l < r):
              mid = int((l + r) / 2)
              if (abs(x - arr[mid]) > abs(x - arr[mid + 1]) or (arr[mid]==arr[mid+1] and arr[mid] < x) ):
                  l = mid + 1
              else:
                  r = mid
  
          # 双指针从中间向两边遍历
          a,b = r,r
          k -= 1
          while(k>0):
              if a == 0:
                  b += 1
                  k -= 1
              elif b == length-1:
                  a -= 1
                  k -= 1
  
              elif(abs(x-arr[a-1]) <= abs(x-arr[b+1])):
                  a -= 1
                  k -= 1
  
              else:
                  b += 1
                  k -= 1
          return arr[a:b+1]
  ```

**思路2：两侧删除**

由于最后目标是保留k个最接近`x`的数，一个很直观的方法就是，从两侧删除离`x`比较远的数，删去`n-k`个，剩下的就是答案！

```python
# Delete the the farther one of leftmost side and rightmost side element until k left
class Solution(object):
    def findClosestElements(self, arr, k, x):
        nums = deque()
        for i in arr:
            nums.append(i)

        while (len(nums) > k):
            if (abs(nums[0] - x) > abs(nums[-1] - x)):
                nums.popleft()
            else:
                nums.pop()

        res = []
        for i in nums:
            res.append(i) 
        return res
```

*说明*：如果直接使用列表删除，复杂度较高，因为每次删除都是`O(n)`复杂度，所以应该转化为双端队列进行删除！

**思路3：二分找k**

由于需要保留的是`k`个连续结果，所以把连续`k`个数看成整体`I`，比较整体的第一个数`I[0]`和`I`后的第一个数`t`与最优结果的距离：

- 若满足`I<=t` 则更新`r`; 否则更新`l`；显然，最后结果在`l`。

  ```python
  class Solution:
      def findClosestElements(self, arr, k, x):
          left = 0
          right = len(arr) - k - 1
          while (left <= right) :
              mid = left+right >> 1
              if (x - arr[mid] > arr[mid + k] - x) :
                  left = mid + 1
              else :
                  right = mid - 1
          return arr[left : left + k]
  ```

  红蓝划分版本的这个形式，还需要进一步思考待定！



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

**解决思路：动态规划** 

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

**进阶思路：动态规划 + 二分**

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

#### problem 410  **Hard**

> 给定一个非负整数数组 `nums` 和一个整数&nbsp;`m` ，你需要将这个数组分成&nbsp;`m`_&nbsp;_个非空的连续子数组。
>
> 设计一个算法使得这&nbsp;`m`_&nbsp;_个子数组各自和的最大值最小。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**nums = [7,2,5,10,8], m = 2
> **输出：**18
> **解释：**
> 一共有四种方法将 nums 分割为 2 个子数组。 
> 其中最好的方式是将其分为 [7,2,5] 和 [10,8] 。
> 因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。</pre>
> **提示：**
>
> *   1 &lt;= nums.length &lt;= 1000
> *   0 &lt;= nums[i] &lt;= 10<sup>6</sup>
> *   1 &lt;= m &lt;= min(50, nums.length)

**思路一：动态规划**

​		「将数组分割为 `m` 段，求……」是动态规划题目常见的问法。

​		`Time_limit`！但是时此类问题的最常见的解法。

显然一维opt无法进行状态定理，可以拓展到二维`opt[i][j]`。

定义： `opt[i][j]`: 记录以`i `结尾的序列的序列分割成`j`段的最优值，即其连续子数组和的最小值。

状态更新：

*白话理解*：对于`opt[5][3]`，显然应该拆分成2段`opt[k][2]`和一段`sum(nums[k+1:i])`三段序列，所以需要遍历所有`k`

> 对于计算任意`opt[i][j]`，即如何形成`j`段：枚举`k`，前`k`个数分成`j-1`段，最后`nums[k+1,i]` 形成第`j`段。
>
> 这`j`段里的最大值，是`max( opt[k][j-1], sum(nums[k+1:i]) )`。
>
> 最优值就是找到枚举过程中的最优值（最小化枚举结果！）

$$
\text{opt}[i][j]=\left\{
\begin{array}{l}
0,\quad & i=j=0 \\
\text{MAXN} & i<j   或 j=0,i\ne0 \\
\min \{\max( opt[k][j-1], \sum(\text{nums}[k+1:i]) ) \}, &其他
\end{array}  

\right.
$$

```python
class dp_Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        n = len(nums)
        f = [[10 ** 9] * m for _ in range(n)]
        # 初始化第一列
        total, sub =0, []
        for i in range(0, n):
            total += nums[i]
            sub.append(total)
            f[i][0] = total
        # 更新
        for j in range(1, m):
            for i in range(j, n):
                minx = 10 ** 9
                for k in range(i):
                    minx = min(minx, max(f[k][j - 1], sub[i]-sub[k]))
                f[i][j] = minx
        return f[n-1][m-1]
```

时间复杂度：$O(n^2 \times m)$，其中 `n` 是数组的长度，`m` 是分成的非空的连续子数组的个数。总状态数为 $O(n \times m)$，状态转移时间复杂度 `O(n)`，所以总时间复杂度为 $O(n^2 \times m)$



**思路二：贪心+二分**

「使……最大值尽可能小」是二分搜索题目常见的问法。

- 求最大值最小，明牌告诉你要用二分，因为是最小，所以要用`>= x`的这个模板

[核心参考](https://leetcode.cn/problems/split-array-largest-sum/solution/bai-hua-er-fen-cha-zhao-by-xiao-yan-gou/)

原题已知一个数组`arr`，需要将其切分为`m`段，使得`m`段中每段和的最大值最小！

> 转换描述：求一个目标值`x`，使得`m`段的各自和的最大值都`<=x`；

- 显然，由于数组里的数不可切分，`x`的范围是`[max(arr), sum(arr)]`

- 搜索最优值`x`的过程可以是二分的！

- 检验`x`是否满足要求：

  以`x`为上限，切分数组`arr`，由于数组切分是有序的，该过程可以贪心的进行！

  如果切分的段数 `cnt<=m`，则满足；否则，不满足。 

- 二分模板分析：

  - 假设最优目标值是18，则在搜索`[16,17,18,19,20]`过程中：
  - 显然对应于`>=18` 是正确的结果。17不满足，在右侧；19满足，在左侧；18满足，在左侧；
  - 对应模板`>=x` 的最小值。

```python
class Solution:
    def splitArray(self, nums: List[int], m: int) -> int:
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt, tot = 0, 0
            for num in nums:
                tot += num
                if tot > x:
                    cnt += 1
                    tot = num
            return True if cnt + 1 <= m else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = max(nums)-1, sum(nums)+1
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```

#### LCP 12

> 为了提高自己的代码能力，小张制定了 `LeetCode` 刷题计划，他选中了 `LeetCode` 题库中的 `n` 道题，编号从 `0` 到 `n-1`，并计划在 `m` 天内**按照题目编号顺序**刷完所有的题目（注意，小张不能用多天完成同一题）。
>
> 在小张刷题计划中，小张需要用 `time[i]` 的时间完成编号 `i` 的题目。此外，小张还可以使用场外求助功能，通过询问他的好朋友小杨题目的解法，可以省去该题的做题时间。为了防止“小张刷题计划”变成“小杨刷题计划”，小张每天最多使用一次求助。
>
> 我们定义 `m` 天中做题时间最多的一天耗时为 `T`（小杨完成的题目不计入做题总时间）。请你帮小张求出最小的 `T`是多少。
>
> **示例 1：**
>
> > 输入：`time = [1,2,3,3], m = 2`
> >
> > 输出：`3`
> >
> > 解释：第一天小张完成前三题，其中第三题找小杨帮忙；第二天完成第四题，并且找小杨帮忙。这样做题时间最多的一天花费了 3 的时间，并且这个值是最小的。
>
> **限制：**
>
> *   1 &lt;= time.length &lt;= 10^5
> *   1 &lt;= time[i] &lt;= 10000
> *   1 &lt;= m &lt;= 1000

**思路：二分+贪心**

本题思路与problem 410 基本一致。

- 转换：求一个目标值`x`，使得每天以`x`未上限进行刷题，最终完成所有题目，所需天数`<=m`即可。
- 二分模板：显然，对于最优目标值`x`，所有`>=x` 的结果都是满足要求的。对应`>=x`模板
- check ：由于每天可以请外援，所以在划分数组时，总是选择当前区间内的最大值移除，统计该区间内其他值之和`tot`，只要其结果`tot <= x`，则可以继续做题(遍历下一个time)

实现：

```python
class Solution:
    def minTime(self, time: List[int], m: int) -> int:
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt, tot, rm = 0, 0, 0
            for num in time:
                if num > rm:
                    tot += rm
                    rm = num
                else:
                    tot += num
                if tot > x:
                    cnt += 1
                    tot, rm = 0, num
            return True if cnt + 1 <= m else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = -1, sum(time)+1
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```



#### problem 1760  **Hard**

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

**思路转化**

求一个目标值`x`，使得数组里每一个数都`<=x`；如果数组里某个数`>x`，则将其拆分为`<=x`的若干个数，要求总共拆分次数`<=maxOperations`即可。

- 二分模板

  显然，对于最小开销，即最优目标值`x`，所有`>=x` 的结果都是满足要求的。

  对应`>=x`模板

- check分析

  对于`>x`的数，尽可能将其拆分成`x`，以减少拆分次数

```python
class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt = 0
            for num in nums:
                if num > x:
                    cnt += (num-1) // x
            return True if cnt <= maxOperations else False
        # 搜索最大子序列和x 使得对原数组中的数拆分次数 <=m
        # 利用红蓝树实现
        l, r = 0, max(nums)+1
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```



#### problem 875

> 珂珂喜欢吃香蕉。这里有 `n` 堆香蕉，第 `i` 堆中有&nbsp;`piles[i]`&nbsp;根香蕉。警卫已经离开了，将在 `h` 小时后回来。
>
> 珂珂可以决定她吃香蕉的速度 `k` （单位：根/小时）。每个小时，她将会选择一堆香蕉，从中吃掉 `k` 根。如果这堆香蕉少于 `k` 根，她将吃掉这堆的所有香蕉，然后这一小时内不会再吃更多的香蕉。&nbsp;&nbsp;
>
> 珂珂喜欢慢慢吃，但仍然想在警卫回来前吃掉所有的香蕉。
>
> 返回她可以在 `h` 小时内吃掉所有香蕉的最小速度 `k`（`k` 为整数）。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**piles = [3,6,7,11], h = 8
> **输出：**4
> </pre>
>
> **示例 2：**
>
> <pre>**输入：**piles = [30,11,23,4,20], h = 5
> **输出：**30
> </pre>
>
> **示例 3：**
>
> <pre>**输入：**piles = [30,11,23,4,20], h = 6
> **输出：**23
> </pre>
>
> &nbsp;
>
> **提示：**
>
> *   1 &lt;= piles.length &lt;= 10<sup>4</sup>
> *   piles.length &lt;= h &lt;= 10<sup>9</sup>
> *   1 &lt;= piles[i] &lt;= 10<sup>9</sup>

思路分析：与problem 1760基本一致，只需要统计，当前速度`x`下，需要多少小时吃完所有香蕉。

```python
class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt = 0
            for num in piles:
                cnt += (num-1) // x + 1
            return True if cnt <= h else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, max(piles)
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```

#### problem 1552

> 在代号为 C-137 的星球上，Rick 发现如果他将两个球放在他新发明的篮子里，它们之间会形成特殊形式的磁力。Rick 有&nbsp;`n`&nbsp;个空的篮子，第&nbsp;`i`&nbsp;个篮子的位置在&nbsp;`position[i]`&nbsp;，Morty&nbsp;想把&nbsp;`m`&nbsp;个球放到这些篮子里，使得任意两球间&nbsp;**最小磁力**&nbsp;最大。
>
> 已知两个球如果分别位于&nbsp;`x`&nbsp;和&nbsp;`y`&nbsp;，那么它们之间的磁力为&nbsp;`|x - y|`&nbsp;。
>
> 给你一个整数数组&nbsp;`position`&nbsp;和一个整数&nbsp;`m`&nbsp;，请你返回最大化的最小磁力。
>
> &nbsp;
>
> **示例 1：**
>
> ![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/08/16/q3v1.jpg)
>
> <pre>**输入：**position = [1,2,3,4,7], m = 3
> **输出：**3
> **解释：**将 3 个球分别放入位于 1，4 和 7 的三个篮子，两球间的磁力分别为 [3, 3, 6]。最小磁力为 3 。我们没办法让最小磁力大于 3 。
> </pre>
>
> **示例 2：**
>
> <pre>**输入：**position = [5,4,3,2,1,1000000000], m = 2
> **输出：**999999999
> **解释：**我们使用位于 1 和 1000000000 的篮子时最小磁力最大。
> </pre>
>
> &nbsp;
>
> **提示：**
>
> *   `n == position.length`
> *   `2 &lt;= n &lt;= 10^5`
> *   `1 &lt;= position[i] &lt;= 10^9`
> *   所有&nbsp;`position`&nbsp;中的整数 **互不相同**&nbsp;。
> *   `2 &lt;= m &lt;= position.length`

**解题思路一： DP**

和[^problem 410 ]比较，两者可以采用相同的模式进行dp分析：

- 定义： `opt[i][j]`: 记录以`i `结尾的序列的序列放入入`j+2`个球（因为球至少2个，忽略了0和1），球间距最小值的最优结果（最大化最小球间距）。

- 状态转移：对于任意的`opt[i][j]`更新而言，枚举`k`，在第`k`位置插入新的球；

  相当于`position[0:k]`分根成`j+1`个球，其最优结果是`opt[k][j-1]`，

  插入球形成了最后一个新间距是`position[i]-position[k]` 
  $$
  \text{opt}[i][j]=\left\{
  \begin{array}{l}
  最大值-最小值,\quad & i=[0:n],\;\;=0 \\
  
  \max \{\min( opt[k][j-1], \;position[i]-position[k] \}, &其他
  \end{array}  
  
  \right.
  $$
  

**解题思路二： 二分搜索**

与前面几题相比，本题是略有不同，属于最大化最小间隔。

- 转换思路：找到一个最大的最优目标值`x`；并且能在数组`position`中找到`m`个数，确保其各数之间的距离在`>=x`。

- 模板分析：对于最优目标值`x`，所有`<=x` 的结果都是有效的。对应`<=x`的二分模板

- `check(x)`：

  如何判断当前的`x`是否满足题目要求，这是本题的难点之一。

  技巧是：在二分之前，先将无序数组`position`转为有序数组，复杂度是$O(n \log n)$；

  进一步二分$O(\log n)$ ，检验的复杂度是$O(n )$，整体复杂度也是$O(n \log n)$

  当`position`是有序的时候，可以贪心进行check：第一个求放在最小位置，之后选择满足`x`要求的最近距离放置下一个球。

```python
class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort()
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt, minx = 1, min(position)
            for p in position:
                if p >= minx+x:
                    cnt += 1
                    minx = p
            return True if cnt >= m else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, max(position)+1
        while l+1 < r:
            mid = l+r >> 1
            if not check(mid):
                r = mid
            else:
                l = mid
        return l
```

#### problem 287

> 给定一个包含&nbsp;`n + 1` 个整数的数组&nbsp;`nums` ，其数字都在&nbsp;`[1, n]`&nbsp;范围内（包括 `1` 和 `n`），可知至少存在一个重复的整数。
>
> 假设 `nums` 只有 **一个重复的整数** ，返回&nbsp;**这个重复的数** 。
>
> 你设计的解决方案必须 **不修改** 数组 `nums` 且只用常量级 `O(1)` 的额外空间。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**nums = [1,3,4,2,2]
> **输出：**2
> </pre>
> **示例 2：**
>
> <pre>**输入：**nums = [3,1,3,4,2]
> **输出：**3
> </pre>
> **提示：**
>
> *   `1 &lt;= n &lt;= 10<sup>5</sup>`
> *   `nums.length == n + 1`
> *   `1 &lt;= nums[i] &lt;= n`
> *   `nums` 中 **只有一个整数** 出现 **两次或多次** ，其余整数均只出现 **一次**
>
> **进阶：**
>
> *   如何证明 `nums` 中至少存在一个重复的数字?
> *   你可以设计一个线性级时间复杂度 `O(n)` 的解决方案吗？

**思路分析：**

本题的难点是，不能修改数组，且无法使用额外空间！

一个最常规的思路是快排，`arr[i]==arr[i+1]`就是所求结果。但是不能修改数组，则需要采用二分搜素。

思路转换：搜索一个目标`x`，统计`<=x`数的个数，如果其统计和超过`x`,证明所求结果在`[0:x]`之间。

模板：显然，所有`>=x` 都是正确的，属于该模板。

```python
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        # 检验当前x：统计比<x 的数的个数
        def check(x):
            cnt = 0
            for num in nums:
                if num <= x:
                    cnt += 1
            return True if cnt > x else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, n
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```

本题还有$O(n)$ 的方法，该方式待补充！



#### problem 1283

> 给你一个整数数组&nbsp;`nums` 和一个正整数&nbsp;`threshold` &nbsp;，你需要选择一个正整数作为除数，然后将数组里每个数都除以它，并对除法结果求和。
>
> 请你找出能够使上述结果小于等于阈值&nbsp;`threshold`&nbsp;的除数中 **最小** 的那个。
>
> 每个数除以除数后都向上取整，比方说 7/3 = 3 ， 10/2 = 5 。
>
> 题目保证一定有解。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**nums = [1,2,5,9], threshold = 6
> **输出：**5
> **解释：**如果除数为 1 ，我们可以得到和为 17 （1+2+5+9）。
> 如果除数为 4 ，我们可以得到和为 7 (1+1+2+3) 。如果除数为 5 ，和为 5 (1+1+1+2)。
> </pre>
>
> **示例 2：**
>
> <pre>**输入：**nums = [2,3,5,7,11], threshold = 11
> **输出：**3
> </pre>
> **示例 3：**
>
> <pre>**输入：**nums = [19], threshold = 5
> **输出：**4
> </pre>
>
> &nbsp;
>
> **提示：**
>
> *   1 &lt;= nums.length &lt;= 5 * 10^4
> *   1 &lt;= nums[i] &lt;= 10^6`
> *   nums.length &lt;=&nbsp;threshold &lt;= 10^6

**思路分析**

本题的目标是找到一个最优目标值`x`，使得数组中每一个数除以`x`以后求和结果`<=threshold`。

显然，可以用二分查找。其中，`x`的范围是`[1,max(nums)]`。

- 二分模板：题目已经明确说了找到最小的`x`，显然任何一个`>=x` 的值都是可行解，对应于`>=x`模板。

```python
class Solution:
    def smallestDivisor(self, nums: List[int], threshold: int) -> int:
        n = len(nums)
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            tot = 0
            for num in nums:
                tot += (num-1)//x + 1
            return True if tot <= threshold else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, max(nums)
        while l+1 < r:
            mid = l+r >> 1
            if check(mid):
                r = mid
            else:
                l = mid
        return r
```

#### problem 1898

> 给你两个字符串 `s` 和 `p` ，其中 `p` 是 `s` 的一个 **子序列** 。同时，给你一个元素 **互不相同** 且下标 **从 0 开始** 计数的整数数组&nbsp;`removable` ，该数组是 `s` 中下标的一个子集（`s` 的下标也 **从 0 开始** 计数）。
>
> 请你找出一个整数 `k`（`0 &lt;= k &lt;= removable.length`），选出&nbsp;`removable` 中的 **前** `k` 个下标，然后从 `s` 中移除这些下标对应的 `k` 个字符。整数 `k` 需满足：在执行完上述步骤后， `p` 仍然是 `s` 的一个 **子序列** 。更正式的解释是，对于每个 `0 &lt;= i &lt; k` ，先标记出位于 `s[removable[i]]` 的字符，接着移除所有标记过的字符，然后检查 `p` 是否仍然是 `s` 的一个子序列。
>
> 返回你可以找出的 **最大**_ _`k`_ _，满足在移除字符后_ _`p`_ _仍然是 `s` 的一个子序列。
>
> 字符串的一个 **子序列** 是一个由原字符串生成的新字符串，生成过程中可能会移除原字符串中的一些字符（也可能不移除）但不改变剩余字符之间的相对顺序。
>
> &nbsp;
>
> **示例 1：**
>
> <pre>**输入：**s = "abcacb", p = "ab", removable = [3,1,0]
> **输出：**2
>
> **解释：**在移除下标 3 和 1 对应的字符后，"a**b**c**a**cb" 变成 "accb" 。
>
> "ab" 是 "**a**cc**b**" 的一个子序列。
> 如果移除下标 3、1 和 0 对应的字符后，"**ab**c**a**cb" 变成 "ccb" ，那么 "ab" 就不再是 s 的一个子序列。
> 因此，最大的 k 是 2 。
>
> **提示：**
>
> *   1 &lt;= p.length &lt;= s.length &lt;= 10<sup>5</sup>`
> *   0 &lt;= removable.length &lt; s.length`
> *   0 &lt;= removable[i] &lt; s.length`
> *   `p` 是 `s` 的一个 **子字符串**
> *   `s` 和 `p` 都由小写英文字母组成
> *   `removable` 中的元素 **互不相同**

- 模板：本题是二分的`<=x`模板，即找到最大的满足题目要求的`x`。

- 细节：进行`check(x)`时，对于移除元素的查找可以使用`set()`，从而实现`O(1)`复杂度查找

```python
class Solution:
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        p += 'I'
        def check(s, x):
            ch = set(removable[0:x])  # 利用set 查找复杂度为O(1)
            cnt = 0
            for i in range(len(s)):
                if s[i] == p[cnt] and i not in ch:
                    cnt += 1
            return True if cnt > len(p)-2 else False
        l, r = -1, len(removable)+1
        while l+1 < r:
            mid = l+r >> 1
            if check(s, mid):
                l = mid
            else:
                r = mid
        return l
```

#### problem 1870 **Hard** <u>?</u>  

> 给你一个浮点数 `hour` ，表示你到达办公室可用的总通勤时间。要到达办公室，你必须按给定次序乘坐 `n` 趟列车。另给你一个长度为 `n` 的整数数组 `dist` ，其中 `dist[i]` 表示第 `i` 趟列车的行驶距离（单位是千米）。
>
> 每趟列车均只能在整点发车，所以你可能需要在两趟列车之间等待一段时间。
>
> *   例如，第 `1` 趟列车需要 `1.5` 小时，那你必须再等待 `0.5` 小时，搭乘在第 2 小时发车的第 `2` 趟列车。
>
> 返回能满足你准时到达办公室所要求全部列车的** 最小正整数 **时速（单位：千米每小时），如果无法准时到达，则返回 `-1` 。
>
> 生成的测试用例保证答案不超过 10<sup>7</sup> ，且 `hour` 的 **小数点后最多存在两位数字** 。
>
> &nbsp;
>
> **示例 1：**
>
> > **输入：**dist = [1,3,2], hour = 6
> > **输出：**1
> > **解释：**速度为 1 时：
> > - 第 1 趟列车运行需要 1/1 = 1 小时。
> > - 由于是在整数时间到达，可以立即换乘在第 1 小时发车的列车。第 2 趟列车运行需要 3/1 = 3 小时。
> > - 由于是在整数时间到达，可以立即换乘在第 4 小时发车的列车。第 3 趟列车运行需要 2/1 = 2 小时。
> > - 你将会恰好在第 6 小时到达。
>
> **<u>易错示例</u> 2：**
>
> > **输入：**dist = [1,1,10], hour = 2.01
> > **输出：**3
> > **解释：**速度为 3 时：
> > - 第 1 趟列车运行需要 1/3 = 0.33333 小时。
> > - 由于不是在整数时间到达，故需要等待至第 1 小时才能搭乘列车。第 2 趟列车运行需要 3/3 = 1 小时。
> > - 由于是在整数时间到达，可以立即换乘在第 2 小时发车的列车。第 3 趟列车运行需要 2/3 = 0.66667 小时。
> > - 你将会在第 2.66667 小时到达。

- 模板分析：本题求最小值，显然对应于`>=x`模板。

- **难点说明**：浮点数的比较大小。

  如易错示例2，对于直接利用python取整计算`math.ceil(10/(2.01-2))=1001`而非目标值`1000`

  如何直接修正取整计算问题，目前尚无可靠思路[^问题标记]

  一个合理的解决方案：[参考](https://leetcode.cn/problems/minimum-speed-to-arrive-on-time/solution/5764zhun-shi-dao-da-de-lie-che-zui-xiao-phes3/)

  - 对最后一位特殊处理：`check(x)`时，对前`n-1`位正常取整，最后只算其浮点数结果，进而与目标值`hour`进行比较。

  - `x`范围变化：由于题目中保证了`hour`最多两位小数，细分度为`0.01`；

    所以正确结果`x`一定属于`[1, max(dist)*100]`，即二分的搜索区间。

- python整数相除取整算法：

  向下取整：`被除数 // 除数`

  向上取整：`(被除数 + 除数 - 1) // 除数`

#### problem 1482

> 给你一个整数数组 `bloomDay`，以及两个整数 `m` 和 `k` 。
>
> 现需要制作 `m` 束花。制作花束时，需要使用花园中 **相邻的 `k` 朵花** 。
>
> 花园中有 `n` 朵花，第 `i` 朵花会在 `bloomDay[i]` 时盛开，**恰好** 可以用于 **一束** 花中。
>
> 请你返回从花园中摘 `m` 束花需要等待的最少的天数。如果不能摘到 `m` 束花则返回 **-1** 。
>
> **示例 1：**
>
> > **输入：**bloomDay = [7,7,7,7,12,7,7], m = 2, k = 3
> > **输出：**12
> > **解释：**要制作 2 束花，每束需要 3 朵。
> > 花园在 7 天后和 12 天后的情况如下：
> > 7 天后：[x, x, x, x, _, x, x]
> > 可以用前 3 朵盛开的花制作第一束花。但不能使用后 3 朵盛开的花，因为它们不相邻。
> > 12 天后：[x, x, x, x, x, x, x]
> > 显然，我们可以用不同的方式制作两束花。
>
>
> 
>
> **提示：**
>
> *   bloomDay.length == n
> *   1 &lt;= n &lt;= 10^5
> *   1 &lt;= bloomDay[i] &lt;= 10^9
> *   1 &lt;= m &lt;= 10^6
> *   1 &lt;= k &lt;= n

**思路分析**：

本题是求解一个最优值`x`使得，在数组中连续`<=x` 的序列串`>=m`。

- 二分模板：对于一个最优`x`，显然所有`>=x` 的值都是正确的，对应于`>=x`模板

```python
class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        l, r = 0, max(bloomDay)+1
        while l+1 < r:
            mid = l+r >> 1
            cnt, seq = 0, 0
            for day in bloomDay:
                if day <= mid:
                    seq += 1
                else:
                    seq = 0
                if seq == k:
                    cnt += 1
                    seq = 0
            if cnt >= m:
                r = mid
            else:
                l = mid
        return -1 if r == max(bloomDay)+1 else r
```

#### problem 1818 **Hard**

> 给你两个正整数数组 `nums1` 和 `nums2` ，数组的长度都是 `n` 。
>
> 数组 `nums1` 和 `nums2` 的 **绝对差值和** 定义为所有 `|nums1[i] - nums2[i]|`（0 &lt;= i &lt; n）的 **总和**（**下标从 0 开始**）。
>
> 你可以选用 `nums1` 中的 **任意一个** 元素来替换 `nums1` 中的 **至多** 一个元素，以 **最小化** 绝对差值和。
>
> 在替换数组 `nums1` 中最多一个元素 **之后** ，返回最小绝对差值和。因为答案可能很大，所以需要对 10<sup>9</sup> + 7 **取余 **后返回。
>
> 
>
> **示例 1：**
>
> > **输入：**nums1 = [1,7,5], nums2 = [2,3,5]
> > **输出：**3
> > **解释：**有两种可能的最优方案：
> >
> > - 将第二个元素替换为第一个元素：[1,**7**,5] =&gt; [1,**1**,5] ，或者
> > - 将第二个元素替换为第三个元素：[1,**7**,5] =&gt; [1,**5**,5]
> > 两种方案的绝对差值和都是 `|1-2| + (|1-3| 或者 |5-3|) + |5-5| = `3
>
> **示例 2：**
>
> > **输入：**nums1 = [1,10,4,4,2,7], nums2 = [9,3,5,1,7,4]
> > **输出：**20
> > **解释：**将第一个元素替换为第二个元素：[**1**,10,4,4,2,7] =&gt; [**10**,10,4,4,2,7]
> > 绝对差值和为 `|10-9| + |10-3| + |4-5| + |4-1| + |2-7| + |7-4| = 20`
>
>
> &nbsp;
>
> **提示：**
>
> *   n == nums1.length
> *   n == nums2.length
> *   1 &lt;= n &lt;= 10<sup>5</sup>
> *   1 &lt;= nums1[i], nums2[i] &lt;= 10<sup>5</sup>

**解决思路**：

由于只能替换`nums1`中的一个数，并在替换该数过程中，实现数组绝对值之差最小化；显然，应该选择替换后可以获得最大改变的一组`nums1[i]`和`nums2[i]`进行替换。

- <u>核心思路</u>：遍历所有`i`，找到进行替换可以获得最大收益的一项。
  - 最大收益即可以最大程度降低`nums1[k]`和`nums2[k]`之间的差。
  - 举例，对于`1`和`7`，他们之间的差距是6，如果在第一个数组中可以找到一个新的数`5`，替换掉`1`，则差距变为2，那么收益就是`6-2=4`。
  - 找这个最大的差异可以用排序好的辅助数组`new1`；在枚举`i`时，在`new1`中找到最接近`nums2[i]`的数，显然采用这个最接近的数替换原来的`nums1[i]`可以获得当前项`i`的最佳收益。
  - 遍历时，再增加一个辅助变量`max_change`记录当前为止的最佳收益。以确保收益始终最大！

```python
class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        ns = sorted(nums1)
        tot, max_change = 0, 0
        for i in range(n):
            diff = abs(nums2[i]-nums1[i])
            tot += diff
            if not diff:
                continue
            l, r = -1, n
            while l + 1 < r:
                mid = l + r >> 1
                if ns[mid] <= nums2[i]:
                    l = mid
                else:
                    r = mid
            if l != -1: # 检查l
                max_change = max(max_change, diff - (nums2[i] - ns[l]))
            if r != n:
                max_change = max(max_change, diff - (ns[r] - nums2[i]))
        mod = (10 ** 9 + 7)
        return (tot-max_change+mod) % mod
```

#### problem 240 **good**

> *二维搜索问题*
>
> 编写一个高效的算法来搜索$m\times n$&nbsp;矩阵 `matrix` 中的一个目标值 `target` 。该矩阵具有以下特性：
>
> *   每行的元素从左到右升序排列。
> *   每列的元素从上到下升序排列。
>
> &nbsp;
>
> **示例 1：**
>
> ![](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/11/25/searchgrid2.jpg)
> > **输入：**matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
> > **输出：**true
>
> **提示：**
>
> *   m == matrix.length
> *   n == matrix[i].length
> *   1 &lt;= n, m &lt;= 300
> *   -10<sup>9</sup>&nbsp;&lt;= matrix[i][j] &lt;= 10<sup>9</sup>
> *   每行的所有元素从左到右升序排列
> *   每列的所有元素从上到下升序排列

思路1：

常规思路是按行或者按列进行二分搜索，复杂度为`O(m log n)`或者`O(n log m)`

实现较为简单：

**思路2**：从右上角开始单调扫描

![](.\images\二维搜索.png)

显然，根据左图可以看到二维数组有一些鲜明的特征，所以可以快速排除某一行或者某一列。

因此我们可以从整个矩阵的右上角开始枚举，假设当前枚举的数是 x：

- 如果 x 等于target，则说明我们找到了目标值，返回true；
- 如果 x小于target，则 x左边的数一定都小于target，我们可以直接排除当前一整行的数；
- 如果 x 大于target，则 x 下边的数一定都大于target，我们可以直接排除当前一整列的数；

复杂度显然为`O(m+n)`

实现：

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        l, r = 0, n-1
        while l<m and r>-1:
            if matrix[l][r] == target:
                return True
            elif matrix[l][r] < target:
                l += 1
            else:
                r -= 1
        return False
```



#### problem 1838 **good**



注意：如果使用常规二分思路，本题的`check(x)`比较难算，且复杂度较高，无法在$O(N)$时间内完成。

思路：排序+滑动窗口

[参考：](https://leetcode.cn/problems/frequency-of-the-most-frequent-element/solution/1838-zui-gao-pin-yuan-su-de-pin-shu-shua-ub57/)

<img src=".\images\Snipaste_2022-05-18_12-23-31.png" style="zoom:50%;" />

首先对原始数据排序，可操作性次数`k`相当于图中的颜色部分面积和；使用双指针始终维持该面积`<=k`，只需要从头到位遍历一次，并求出过程中`l-r+1`的最大值，就是本题的目标值。

- 增加面积按行实现
- 减少面积是按列的

```python
class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        tot, ans = 0, 1
        l, r = 0, 1
        while r<n:
            tot += (nums[r]-nums[r-1]) * (r-l)   # 按行增加面积
            while tot > k:
                tot -= (nums[r]-nums[l])    # 按列介绍面积
                l += 1
            ans = max(ans, r-l+1)
            r += 1
        return ans
```

#### problem 540

> 给你一个仅由整数组成的有序数组，其中每个元素都会出现两次，唯有一个数只会出现一次。
>
> 请你找出并返回只出现一次的那个数。
>
> 你设计的解决方案必须满足 `O(log n)` 时间复杂度和 `O(1)` 空间复杂度。
>
> &nbsp;
>
> **示例 1:**
>
> > **输入:** nums = [1,1,2,3,3,4,4,8,8]
> > **输出:** 2

思路：
初始时，二分查找的左边界是 00，右边界是数组的最大下标。每次取左右边界的平均值 mid 作为待判断的下标，根据 mid 的奇偶性决定和左边或右边的相邻元素比较：

- 如果mid 是偶数，则比较`nums[mid]` 和`nums[mid+1]` 是否相等；


- 如果 mid 是奇数，则比较`nums[mid−1] `和 `nums[mid] `是否相等。

**技巧：**

对于分奇数偶数讨论的问题，可以利用`按位异或`解决。

- 如果mid 是偶数，$mid +1 = mid \bigotimes 1$

- 如果 mid 是奇数，$mid-1 = mid \bigotimes 1$

```python
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n-1
        while l < r:
            mid = l+r >> 1
            if nums[mid]==nums[mid^1]:
                l = mid+1
            else:
                r = mid
        return nums[l]
```

## 第三章：图论

### 1，dfs、bfs

#### A，算法模板

##### **dfs+栈实现：**

- 在python中，可以使用列表`list`实现`栈`、`队列`两种数据结构。

  - 实现`stack`：`a.append(s),    a.pop()`
  - 实现`queue`：`a.insert(0, s), a.pop()`
  - 对于复杂场景，建议使用`collections.deque`； *注意：*LeetCode编译器已导入`collections`包

- <u>技巧</u>：防止重复统计

  有两种方式可以实现防止重复统计，一是提前修改已访问标志，再加入栈；二是在每次统计前判断是否已修改。

- **推荐实现1**：确保每个位置只访问一次

  `实现1`的核心在于`# 入栈立即修改已访问标志`和`# 判断可访问条件`

  ```python
  if grid[i][j] == target:
      a = [(i, j)]    
      grid[i][j] = new	# 入栈立即修改已访问标志
      while a:
          x, y = a.pop()
          for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
          	if 0 <= ii < m and 0 <= jj < n and grid2[ii][jj] == target:	# 判断可访问条件
                  a.append((ii, jj))                
                  grid[ii][jj] = new	# 入栈立即修改已访问标志
  ```
  
- 常规实现2：统计前判断

  ```python
  if grid[i][j] == target:
      stack = [(i, j)]
      while stack:
          cur_i, cur_j = stack.pop()
          if cur_i < 0 or cur_j < 0 or cur_i == len(grid) or cur_j == len(grid[0]) \
                  or grid[cur_i][cur_j] != 1:
              continue					# 统计前判断，不符合条件则跳过不统计
          cur += 1
          grid[cur_i][cur_j] = new
          for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
              next_i, next_j = cur_i + di, cur_j + dj
              stack.append((next_i, next_j))
      ans = max(ans, cur)
  ```

##### **bfs+队列实现:**

有了dfs栈实现的基础，bfs过程就只需要将其中的数据结构stack 改成 `collections.deque`即可！

- `collections.deque`说明
  - `deque(a)`初始化要求`a`必须可迭代，如`a=[1]`可行
  - 和`list`类似，`append(num)`、`pop()`都只对末端处理。
  - 在首端插入用`insert(index, num)`，首端出队列用`popleft()`
  - 先进先出队列采用`append()`、`popleft()`组合实现

```python
if grid[i][j] == target:
    q = collections.deque([(i, j)])    
    grid[i][j] = new	# 入队列立即修改已访问标志
    while a:
        x, y = a.popleft()
        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
        	if 0 <= ii < m and 0 <= jj < n and grid2[ii][jj] == target:	# 判断可访问条件
                a.append((ii, jj))                
                grid[ii][jj] = new	# 入队列立即修改已访问标志
```



##### **dfs 递归实现：**

实现递归的关键有：

- <u>递归终止条件</u>

  对于本题而言，将不满足要求的情况设置为终止，并返回 0。

- 递归的结果一般都以`ans += bfs()` 呈现，以此计算整个过程的所有结果。

```python
class Solution:
    def bfs(self, grid, x, y, target, new):
        m, n = len(grid), len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or grid[x][y]!=target: # 递归终止条件，核心难点！
            return 0 
        ans = 1
        grid[x][y] = new
        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
            ans += self.bfs(grid, ii, jj, target, new)
        return ans
    
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # dfs
        target, new = 1, 0
        m, n = len(grid), len(grid[0])
        maxn = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == target:
                    maxn = max(maxn, self.bfs(grid, i, j, target, new))
        return maxn
```

*说明*：递归算法思路比较清晰，但是递归过程往往耗费时间空间都较大，而且递归理解难度更大，所以并不推荐写递归`dfs`;更值得研究也更简单的是通过`栈`实现，而且将其改为`队列`后，可以快速实现`bfs`!

##### 常用技巧：

- **添加flag**

  bfs和dfs的简单题中，常常会出现有条件的统计，提个有效的解决方案是添加判断标志`flag`，用于判断相关结果是否需要统计。

  详见`p1020`、`p1905`、`p1254`等。

- **bfs添加计数**

  在计算利用bfs扩散的次数时，可以增加一个bfs层数计数器，直接绑定添加到队列中！

  详见`p1162`

#### B, 练习题目

##### problem 733

> 有一幅以&nbsp;`m x n`&nbsp;的二维整数数组表示的图画&nbsp;`image`&nbsp;，其中&nbsp;`image[i][j]`&nbsp;表示该图画的像素值大小。
>
> 你也被给予三个整数 `sr` ,&nbsp; `sc` 和 `newColor` 。你应该从像素&nbsp;`image[sr][sc]`&nbsp;开始对图像进行 上色**填充** 。
>
> 为了完成** 上色工作** ，从初始像素开始，记录初始坐标的 **上下左右四个方向上** 像素值与初始坐标相同的相连像素点，接着再记录这四个方向上符合条件的像素点与他们对应 **四个方向上** 像素值与初始坐标相同的相连像素点，……，重复该过程。将所有有记录的像素点的颜色值改为&nbsp;`newColor`&nbsp;。
>
> 最后返回 _经过上色渲染后的图像&nbsp;_。
>
> &nbsp;
>
> **示例 1:**
>
> ![](.\images\flood1-grid.jpg)
>
> > **输入:** image = [[1,1,1],[1,1,0],[1,0,1]]，sr = 1, sc = 1, newColor = 2
> > **输出:** [[2,2,2],[2,2,0],[2,0,1]]
> > **解析:** 在图像的正中间，(坐标(sr,sc)=(1,1)),在路径上所有符合条件的像素点的颜色都被更改成2。
> > 注意，右下角的像素没有更改为2，因为它不是在上下左右四个方向上与初始点相连的像素点。

思路分析：

本题是最基础的图的遍历问题，可以采用通用模板实现。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        old = image[sr][sc]
        if newColor == old:  # 防止陷入死循环
            return image
        res = [(sr, sc)]
        while res:
            l, r = res.pop()
            image[l][r] = newColor
            for i, j in [(l, r-1),(l-1, r),(l, r+1), (l+1, r)]:
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old:
                    res.append((i,j))
        return image
```



problem 200

> 给你一个由&nbsp;`'1'`（陆地）和 `'0'`（水）组成的的二维网格，请你计算网格中岛屿的数量。
>
> 岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。
>
> 此外，你可以假设该网格的四条边均被水包围。
>
> &nbsp;
>
> **示例 1：**
>
> > **输入：**grid = [
> >   ["1","1","1","1","0"],
> >   ["1","1","0","1","0"],
> >   ["1","1","0","0","0"],
> >   ["0","0","0","0","0"]
> > ]
> > **输出：**1
>
> **提示：**
>
> *   1 &lt;= m, n &lt;= 300

思路分析：一次遍历+一次dfs

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # dfs
        old, newColor= "1", "0"
        m, n = len(grid), len(grid[0])
        
        res = 0
        for x in range(m):
            for y in range(n):
                if grid[x][y] == old:
                    res += 1
                    a = [(x, y)]
                    while a:
                        l, r = a.pop()
                        grid[l][r] = newColor
                        for i, j in [(l, r - 1), (l - 1, r), (l, r + 1), (l + 1, r)]:
                            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == old:
                                a.append((i, j))

        return res
```



problem 695

> 给你一个大小为 `m x n` 的二进制矩阵 `grid` 。
>
> **岛屿**&nbsp;是由一些相邻的&nbsp;`1`&nbsp;(代表土地) 构成的组合，这里的「相邻」要求两个 `1` 必须在 **水平或者竖直的四个方向上 **相邻。你可以假设&nbsp;`grid` 的四个边缘都被 `0`（代表水）包围着。
>
> 岛屿的面积是岛上值为 `1` 的单元格的数目。
>
> 计算并返回 `grid` 中最大的岛屿面积。如果没有岛屿，则返回面积为 `0` 。
>
> &nbsp;
>
> **示例 1：**
>
> > **输入：**grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
> > **输出：**6
> > **解释：**答案不应该是 `11` ，因为岛屿只能包含水平或垂直这四个方向上的 `1` 。

思路分析：遍历+dfs+计数

```python
class Solution:
    def bfs(self, grid, x, y, target, new):
        m, n = len(grid), len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or grid[x][y]!=target:
            return 0
        ans = 1
        grid[x][y] = new
        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
            ans += self.bfs(grid, ii, jj, target, new)
        return ans
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # dfs
        target, new = 1, 0
        m, n = len(grid), len(grid[0])
        maxn = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == target:
                    maxn = max(maxn, self.bfs(grid, i, j, target, new))
        return maxn
```



##### problem 1254

> 二维矩阵 `grid`&nbsp;由 `0`&nbsp;（土地）和 `1`&nbsp;（水）组成。岛是由最大的4个方向连通的 `0`&nbsp;组成的群，封闭岛是一个&nbsp;`完全` 由1包围（左、上、右、下）的岛。
>
> 请返回 _封闭岛屿_ 的数目。
>
> &nbsp;
>
> **示例 1：**
>
> ![](.\images\sample_3_1610.png)
>
> > **输入：**grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
> > **输出：**2
> > **解释：**
> > 灰色区域的岛屿是封闭岛屿，因为这座岛屿完全被水域包围（即被 1 区域包围）。

思路分析：遍历+bfs + flag

注意本题由于在边界上不用计数，所以可以采用一个边界访问标记flag，如果到达边界则忽略置为`False`

```python
class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    a, flag = [(i, j)], True
                    while a:
                        x, y = a.pop()
                        if x==0 or x == m-1 or y==0 or y==n-1:
                            flag = False
                        for ii, jj in zip((x,x-1,x,x+1),(y-1,y,y+1,y)):
                            if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == 0:
                                grid[ii][jj] = 1
                                a.append((ii,jj))
                    if flag:
                        cnt += 1
        return cnt
```



##### problem 1162  **Hard**

> 你现在手里有一份大小为&nbsp;`n x n`&nbsp;的 网格 `grid`，上面的每个 单元格 都用&nbsp;`0`&nbsp;和&nbsp;`1`&nbsp;标记好了。其中&nbsp;`0`&nbsp;代表海洋，`1`&nbsp;代表陆地。
>
> 请你找出一个海洋单元格，这个海洋单元格到离它最近的陆地单元格的距离是最大的，并返回该距离。如果网格上只有陆地或者海洋，请返回&nbsp;`-1`。
>
> 我们这里说的距离是「曼哈顿距离」（&nbsp;Manhattan Distance）：`(x0, y0)` 和&nbsp;`(x1, y1)`&nbsp;这两个单元格之间的距离是&nbsp;`|x0 - x1| + |y0 - y1|`&nbsp;。
>
> &nbsp;
>
> **示例 1：**
>
> **![](.\images\1336_ex2.jpeg)**
>
> > **输入：**grid = [[1,0,0],[0,0,0],[0,0,0]]
> > **输出：**4
> > **解释： **
> > 海洋单元格 (2, 2) 和所有陆地单元格之间的距离都达到最大，最大距离为 4。
>
>
> &nbsp;
>
> **提示：**
>
> *   n == grid.length
> *   n == grid[i].length
> *   1 &lt;= n&nbsp;&lt;= 100
> *   `grid[i][j]`&nbsp;不是&nbsp;`0`&nbsp;就是&nbsp;`1`

**思路1：多源bfs**

这是`Tree的bfs`拓展版：从一个源点出发，每次向`上下左右`4个方向扩散。

- 注意：每次都只能扩散`上下左右`4个方向，这样可以保证每次曼哈顿距离是递增的。

- <u>核心技巧</u>：如果采用扫描每个可行源点，然后`bfs`扩散找到最近的陆地；这样的方法涉及了很多冗余计算！

  [参考](https://leetcode.cn/problems/as-far-from-land-as-possible/solution/jian-dan-java-miao-dong-tu-de-bfs-by-sweetiee/)：

  只要先把所有的陆地都入队，然后从各个陆地同时开始一层一层的向海洋扩散，*那么最后扩散到的海洋就是最远的海洋*！
  下面是扩散的图示，1表示陆地，0表示海洋。每次扩散的时候会标记相邻的4个位置的海洋：

  ![](.\images\ludihaiyang.png)

  > 你可以想象成你从每个陆地上派了很多支船去踏上伟大航道，踏遍所有的海洋。每当船到了新的海洋，就会分裂成4条新的船，向新的未知海洋前进（访问过的海洋就不去了）。如果船到达了某个未访问过的海洋，那他们是第一个到这片海洋的。很明显，这么多船最后访问到的海洋，肯定是离陆地最远的海洋。

  **最后出队列的是就是最优结果，即离陆地最远的海洋！**

  ```python
  class Solution:
      def maxDistance(self, grid: List[List[int]]) -> int:
          n = len(grid)
          a = collections.deque()
          # 所有陆地先入队
          for i in range(n):
              for j in range(n):
                  if grid[i][j]==1:
                      a.append((i,j,0))
          # 在曼哈顿距离下，入队只能是以距离为1 扩散
          ans = -1
          while a:
              x, y, ans = a.popleft()
              for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
                  if 0 <= ii < n and 0 <= jj < n and grid[ii][jj]==0:
                      grid[ii][jj] = 1
                      a.append((ii,jj, ans+1))                    
          return -1 if ans<=0 else ans
  ```


**思路2：多源最短路**

显然，如果将

##### problem 1020

> 给你一个大小为 `m x n` 的二进制矩阵 `grid` ，其中 `0` 表示一个海洋单元格、`1` 表示一个陆地单元格。
>
> 一次 **移动** 是指从一个陆地单元格走到另一个相邻（**上、下、左、右**）的陆地单元格或跨过 `grid` 的边界。
>
> 返回网格中** 无法 **在任意次数的移动中离开网格边界的陆地单元格的数量。
>
> &nbsp;
>
> **示例 1：**
>
> ![](.\images\enclaves1.jpg)
> > **输入：**grid = [[0,0,0,0],[1,0,1,0],[0,1,1,0],[0,0,0,0]]
> > **输出：**3
> > **解释：**有三个 1 被 0 包围。一个 1 没有被包围，因为它在边界上。
>
> **提示：**
>
> *   `m == grid.length`
> *   `n == grid[i].length`
> *   1 &lt;= m, n &lt;= 500

**思路1**

分析：dfs + flag判断边界 + 栈计数

本题是正常的图连通问题，只需要添加`flag`判断该区域是否会连接到边界；再对每次dfs过程的入栈元素计数，即可获得最终解。

实现：

```python
class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    grid[i][j] = 0
                    a = [(i, j)]
                    cnt, flag = 0, True
                    while a:
                        x, y = a.pop()
                        cnt += 1
                        if x==0 or x == m-1 or y==0 or y==n-1:
                            flag = False
                        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
                            if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == 1:
                                grid[ii][jj] = 0
                                a.append((ii, jj,))
                    if flag:
                        ans += cnt
        return ans
```

**思路2：并查集**

并查集的做法是，遍历整个网格，对于网格中的每个陆地单元格，将其与所有相邻的陆地单元格做合并操作。由于需要判断每个陆地单元格所在的连通分量是否和网格边界相连，因此并查集还需要记录每个单元格是否和网格边界相连的信息，在合并操作时更新该信息。

在遍历网格完成并查集的合并操作之后，再次遍历整个网格，通过并查集中的信息判断每个陆地单元格是否和网格边界相连，统计飞地的数量。

```python
class UnionFind:
    def __init__(self, grid: List[List[int]]):
        m, n = len(grid), len(grid[0])
        self.parent = [0] * (m * n)
        self.rank = [0] * (m * n)

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def merge(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.onEdge[x] |= self.onEdge[y]
        elif self.rank[x] < self.rank[y]:
            self.parent[x] = y
            self.onEdge[y] |= self.onEdge[x]
        else:
            self.parent[y] = x
            self.onEdge[x] |= self.onEdge[y]
            self.rank[x] += 1

class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        uf = UnionFind(grid)
        m, n = len(grid), len(grid[0])
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                if v:
                    idx = i * n + j
                    if j + 1 < n and grid[i][j + 1]:
                        uf.merge(idx, idx + 1)
                    if i + 1 < m and grid[i + 1][j]:
                        uf.merge(idx, idx + n)
        return sum(grid[i][j] and not uf.onEdge[uf.find(i * n + j)] for i in range(1, m - 1) for j in range(1, n - 1))
```





##### problem 1905

> 给你两个&nbsp;`m x n`&nbsp;的二进制矩阵&nbsp;`grid1` 和&nbsp;`grid2`&nbsp;，它们只包含&nbsp;`0`&nbsp;（表示水域）和 `1`&nbsp;（表示陆地）。一个 **岛屿**&nbsp;是由 **四个方向**&nbsp;（水平或者竖直）上相邻的&nbsp;`1`&nbsp;组成的区域。任何矩阵以外的区域都视为水域。
>
> 如果 `grid2`&nbsp;的一个岛屿，被 `grid1`&nbsp;的一个岛屿&nbsp;**完全** 包含，也就是说 `grid2`&nbsp;中该岛屿的每一个格子都被 `grid1`&nbsp;中同一个岛屿完全包含，那么我们称 `grid2`&nbsp;中的这个岛屿为 **子岛屿**&nbsp;。
>
> 请你返回 `grid2`&nbsp;中 **子岛屿**&nbsp;的 **数目**&nbsp;。
>
> &nbsp;
>
> **示例 1：**
>
> ![](.\images\test1.png)
> > **输入：**grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]], grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]
> > **输出：**3
> > **解释：**如上图所示，左边为 grid1 ，右边为 grid2 。
> > grid2 中标红的 1 区域是子岛屿，总共有 3 个子岛屿。

*思路1*：dfs + flag

对于`grid2`进行遍历，对于它的每一座岛屿，都用flag判断是否是`grid1`的子岛屿，如果都是，则是；如果有一个不是，则整个岛都不是；

实现：

```python
class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        m, n = len(grid1), len(grid1[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    a = [(i, j)]
                    grid2[i][j] = 0
                    flag = True
                    while a:
                        x, y = a.pop()
                        if grid1[x][y] != 1:
                            flag = False
                        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
                            if 0 <= ii < m and 0 <= jj < n and grid2[ii][jj] == 1:
                                grid2[ii][jj] = 0
                                a.append((ii, jj))
                    if flag:
                        ans += 1
        return ans
```

**思路2：并查集**

### 2，二叉树

#### A, 常见算法

##### 1，二叉树问题

- 前序中序后序遍历：

  二叉树常见遍历方式存在`前序(根左右)，中序(左根右)、后序(左右根)`三种遍历方式

- **二叉树构造生成**

  只要知道`中序`+`前序/后序`的一种遍历结果，便可以重新构建二叉树。

  详见`p105(前中)`、`p106(中后)`

#### B, 练习题目

##### problem 105

> 给定两个整数数组&nbsp;`preorder` 和 `inorder`&nbsp;，其中&nbsp;`preorder` 是二叉树的**先序遍历**， `inorder`&nbsp;是同一棵树的**中序遍历**，请构造二叉树并返回其根节点。
>
> &nbsp;
>
> **示例 1:**
>
> ![](.\images\tree.jpg)
> > **输入：** preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
> > **输出：** [3,9,20,null,null,15,7]

**思路：递归构造二叉树**

- 对于任意一颗树而言，前序遍历的形式总是

  `[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]`

  即根节点总是前序遍历中的第一个节点。而中序遍历的形式总是

  `[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]`

- 递归的思路就是，每次在中序遍历中找到`root`，从而可以将问题拆分成两个子问题：左边构造左子树，右边构造右子树。

- <u>细节 1</u>：

  对于`root`位置查找，如果每次都要扫描查找效率较低，对于无重复元素的数组，可以利用哈希映射`dict`；从而快速找到其索引下标`root_index`

- <u>细节 2</u>：递归终止条件

  但传递的`inorder`数组为空时，即不可继续构造，`return None`即可

- <u>实际优化</u>：

  不用每次构造左右子树都重新传递两个&nbsp;`preorder` 和 `inorder`数组，实际上，可以用指针定位当前传递数据的位置即可。

  再进一步，由于前序遍历最前面的元素往往是当前树的根节点，只要在递归调用时保证`根、左、右`顺序递归，实际上每次对应的根都是当前`preorder`最前面的元素

**实现**：

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        n = len(inorder)
        idx_map = {element: i for i, element in enumerate(inorder)}

        def my_tree(in_l, in_r):
            if in_l > in_r:
                return None
            # preorder第一个数是根，刚好符合二叉树dfs遍历过程
            val = preorder.pop(0)   
            root_index = idx_map[val]

            root = TreeNode(val)
            root.left = my_tree(in_l, root_index-1)
            root.right = my_tree(root_index+1, in_r)
            return root

        return my_tree(0, n-1)
```

#### problem 106

> 与problem 105类似，此时给定的的是中序和后序遍历的结果，构建二叉树，并返回根节点。

**思路：递归实现**

- 与problem 105解法极其类似，需要注意的是，由于后序遍历最后一个数往往是当前的`root`,所以每次总是取出当前`postorder`的最后一个数。
- 进一步，由于后序遍历顺序是`左右根`，所以反向构造时，要先`dfs`其右子树，最后访问左子树。

实现：

```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:        
        n = len(inorder)
        idx_map = {element: i for i, element in enumerate(inorder)}

        def my_tree(in_left, in_right):
            if in_left > in_right:
                return None
            val = postorder.pop()  # 后续遍历
            
           	root_index = idx_map[val]
            root = TreeNode(val)
            root.right = my_tree(root_index+1, in_right)	# 后续遍历，先构造右子树
            root.left = my_tree(in_left, root_index-1)
            return root
        
        return my_tree(0, n-1)
```





## 第四章：动态规划

### dp说明

动态规划方法可以解决绝大多数，满足马尔科夫性的问题：即无后效性，`s(n)`仅仅与`s(n-1)`有关，与之前所有状态无关！

#### `opt`状态定义：

能否成功解决一个dp问题，往往与问题状态定义高度相关。一个正确且合理的定义是解决dp问题的关键。

一般实际解决dp问题，通常采用*自底向上递推*求解。

几个常用步骤：

- <u>排序</u>，如果问题输入数据并不存在序关系，且题目结果也没有要求顺序要求，或者序关系存在实际应用意义，此时都可以先对数据排序，后进行dp求解！
- 分析简单的情景，并确定问题划分方式，即怎么拆开考虑。
- <u>最优状态OPT定义</u>？ 
  - 选择几个参数，以及每个参数含义
  - 考虑参数含义从`0`到`i`、从`i`到`j`、背包问题加最大容量限定`k/w`等
- <u>确定初始条件</u>
- <u>状态转移方程</u>
  - 简单场景：`opt[i]`仅与其前一项`opt[i-1]`有关，或者仅仅和自己有关。
  - 二维场景：对于当前的`opt[i][j]`，常常需要枚举`opt[k][j-1]`或`opt[i-1][k]`
  - 多维场景：一般都是理论价值，实际算法实现中，最多写`3`维，更新过程并不简单！
- OPT中加入*递归终止条件*或者*递推起始条件*

### 股票问题

[参考](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/solution/gu-piao-wen-ti-python3-c-by-z1m/)

#### problem 121

**单次股票买卖问题**

> 给定一个数组 `prices` ，它的第&nbsp;`i` 个元素&nbsp;`prices[i]` 表示一支给定股票第 `i` 天的价格。
>
> 你只能选择 **某一天** 买入这只股票，并选择在 **未来的某一个不同的日子** 卖出该股票。设计一个算法来计算你所能获取的最大利润。
>
> 返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。
>
> &nbsp;
>
> **示例 1：**
>
> > **输入：**[7,1,5,3,6,4]
> > **输出：**5
> > **解释：**在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
> >      注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
>
> **提示：**
>
> *   1 &lt;= prices.length &lt;= 10<sup>5</sup>
> *   0 &lt;= prices[i] &lt;= 10<sup>4</sup>

本题是简单题，只需要记录一下截止目前为之的最小股票价格，就能算的当前最佳收益。

- `opt[i]`到第`i`天能获得的最高收益。
- 状态转移：`dp[i]=max(dp[i−1],prices[i]−minprice)`

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minx = prices[0]
        for i in range(len(prices)):
            minx, prices[i] = min(minx, prices[i]), prices[i]-minx
        return max(prices)
```

