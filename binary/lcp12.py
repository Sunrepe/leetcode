from typing import List
import numpy as np
MAXN = 2e9

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


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.minTime(
            time = [999,999,999], m = 4
        )
    )