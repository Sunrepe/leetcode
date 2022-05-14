from typing import List
import numpy as np
MAXN = 2e9

class Solution:
    def minimumSize(self, nums: List[int], maxOperations: int) -> int:
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt = 0
            for num in nums:
                if num > x:
                    cnt += (num-1) // x
            return True if cnt <= maxOperations else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, max(nums)+1
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
        slu.minimumSize(
            nums = [2,4,8,2], maxOperations = 4
        )
    )