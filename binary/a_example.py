from typing import List
import numpy as np
MAXN = 2e9


class Solution:
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

def IsBlue(m,x):
    return m < x
def binary_search(arr,x):
    N = len(arr)
    l, r = -1, N
    while l+1 != r:
        mid = l+r >> 1
        if IsBlue(mid, x):
            l = mid
        else:
            r = mid



if __name__ == '__main__':

    slu = Solution()
    print(
        slu.lengthOfLIS(
            nums = [10,9,2,5,3,7,101,18]
        )
    )