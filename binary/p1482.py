from typing import List

class Solution:
    def minDays(self, bloomDay: List[int], m: int, k: int) -> int:
        l, r = 0, max(bloomDay)+1
        while l+1 < r:
            mid = l+r >> 1
            cnt, seq = 0, 0
            for day in bloomDay:
                if day <= mid:
                    seq += 1
                else:
                    seq = 0
                if seq == k:
                    cnt += 1
                    seq = 0
            if cnt >= m:
                r = mid
            else:
                l = mid
        return -1 if r == max(bloomDay)+1 else r


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.minDays(
            bloomDay = [7,7,7,7,12,7,7], m = 2, k = 3
        )
    )