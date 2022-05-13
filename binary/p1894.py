from typing import List
import numpy as np
MAXN = 2e9

def pri(i,j,l,nums):
    for t in range(j+1,l+1):
            print(nums[i], nums[j], nums[t])

class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        length = len(chalk)
        total = sum(chalk)
        total = k%total
        # 一次遍历 O(n)难度 还可以进一步简化为 O(log n)
        l=0
        while(total>0):
            total -= chalk[l]
            l += 1
        return l if total==0 else l-1


if __name__ == '__main__':

    slu = Solution()
    print(
        Solution().chalkReplacer(chalk = [5,1,5], k = 22)
    )