from typing import List


class Solution:
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        # 对于最后一个元素的特殊处理
        n = len(dist)
        if hour <= n-1:
            return -1
        l, r = 0, max(dist)*100+1
        while l+1 < r:
            mid = l+r >> 1
            res = 0
            for x in dist[:-1]:
                res += (x+mid-1)//mid
            # 单独计算最后一次用时
            res += dist[-1]/mid
            if res <= hour:
                r = mid
            else:
                l = mid
        return r


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.minSpeedOnTime(
            dist = [1,3,2], hour = 2.7
        )
    )