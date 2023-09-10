# 动态规划入门题
# My_solution: opt[i]记录以s[i]为结尾的最长substring 的长度，需要更多的内存空间：临时的substring记录插入s[i]之前的子串。

from typing import List


class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        opt = []
        tmps = ""
        maxn = 0
        for i, se in enumerate(s):
            idx = tmps.find(se)
            tmps = f"{tmps[idx+1:]}{se}"
            maxn = max(maxn, len(tmps))
        return maxn


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.lengthOfLongestSubstring(
            s = "pwwkew"
        )
    )