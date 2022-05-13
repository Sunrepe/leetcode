from typing import List
import numpy as np
MAXN = 2e9

def pri(i,j,l,nums):
    for t in range(j+1,l+1):
            print(nums[i], nums[j], nums[t])

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

class Solution1:
    def triangleNumber(self, nums: List[int]) -> int:
        length = len(nums)
        nums.sort()
        res = 0
        for i in range(length-2):
            for j in range(i+1,length-1):
                l, r = j+1, length-1
                while(l<r): # 二分搜索上确界upper_bandon
                    mid = int((l+r)/2)
                    if (nums[i]+nums[j] <= nums[mid]):
                        r = mid
                    else: l = mid+1
                if nums[i]+nums[j] > nums[l]:
                    res += (l-j)
                else:
                    res += (l-j-1)
        return res


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.triangleNumber(
        nums = [2,0,0,0,0,0]
        )
    )