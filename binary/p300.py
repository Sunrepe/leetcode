from typing import List
import numpy as np
MAXN = 2e9


class Solution2:
    def lengthOfLIS(self, nums: List[int]) -> int:
        length = len(nums)
        opt = [1]
        # 一次遍历 O(n)难度
        for i in range(1, length):
            # 找到小于当前项的最大值
            maxx = 1
            for _ in range(i):
                if nums[_] < nums[i] and opt[_]+1 > maxx:
                    maxx = opt[_]+1
            opt.append(maxx)
        return max(opt)


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


if __name__ == '__main__':

    slu = Solution2()
    print(
        slu.lengthOfLIS(
            nums = [10,9,2,5,3,7,101,18]
        )
    )