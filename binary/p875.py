from typing import List
import numpy as np
MAXN = 2e9

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
        l, r = 0, max(piles)+1
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
        slu.minEatingSpeed(
            piles = [30,11,23,4,20], h = 6
        )
    )