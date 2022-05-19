from typing import List


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        l, r = -1, n+1
        while l+1 < r:
            mid = l+r >> 1
            if mid == 0 or citations[n-mid] >= mid:
                l = mid
            else:
                r = mid
        return l


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.hIndex(
            citations = [1]
        )
    )