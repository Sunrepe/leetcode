from typing import List
import numpy as np
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

class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        total = sum(nums)
        l = 0
        r = 0
        if total < target:
            return 0
        total = nums[0]
        lennum = len(nums)
        minLen = lennum
        while (l<=r):
            while(total<target and r < lennum-1):
                r += 1
                total += nums[r]
            if total>= target:
                minLen = min(minLen, r-l+1)
            total -= nums[l]
            l += 1
        return minLen

if __name__ == '__main__':
    slu = Solution()
    a = [1,1,1,1,1,1,1]
    b = 11
    print(slu.minSubArrayLen(b,a))