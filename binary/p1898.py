from typing import List


class Solution:
    def maximumRemovals(self, s: str, p: str, removable: List[int]) -> int:
        p += 'I'
        def check(s, x):
            ch = set(removable[0:x])  # 利用set 查找复杂度为O(1)
            cnt = 0
            for i in range(len(s)):
                if s[i] == p[cnt] and i not in ch:
                    cnt += 1
            return True if cnt > len(p)-2 else False
        l, r = -1, len(removable)+1
        while l+1 < r:
            mid = l+r >> 1
            if check(s, mid):
                l = mid
            else:
                r = mid
        return l


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.maximumRemovals(
            s = "abcacb", p = "ab", removable = [3,1,0]
        )
    )