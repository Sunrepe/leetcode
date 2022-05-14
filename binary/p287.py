from typing import List
import numpy as np
MAXN = 2e9

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
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


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.findDuplicate(
            nums = [1,3,4,2,2]
        )
    )