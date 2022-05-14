from typing import List
import numpy as np
MAXN = 2e9


def pri(f):
    for i in range(len(f)):
        t = []
        for j in range(i+1):
            t.append(f[i][j])
        pri(t)

class my_dp_Solution:
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


def check(x, arr, m):
    res, tot = 0, 0
    for num in arr:
        tot += num
        if tot > x:
            res += 1
            tot = num
    return True if res+1 <= m else False


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



if __name__ == '__main__':

    slu = Solution()
    print(
        slu.splitArray(
            nums = [1,4,4], m = 3
        )
    )