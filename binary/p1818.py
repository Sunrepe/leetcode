from typing import List


class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        ns = sorted(nums1)
        tot, max_change = 0, 0
        for i in range(n):
            diff = abs(nums2[i]-nums1[i])
            tot += diff
            if not diff:
                continue
            l, r = -1, n
            while l + 1 < r:
                mid = l + r >> 1
                if ns[mid] <= nums2[i]:
                    l = mid
                else:
                    r = mid
            if l != -1: # 检查l
                max_change = max(max_change, diff - (nums2[i] - ns[l]))
            if r != n:
                max_change = max(max_change, diff - (ns[r] - nums2[i]))
        mod = (10 ** 9 + 7)
        return (tot-max_change+mod) % mod




if __name__ == '__main__':
    slu = Solution()
    print(
        slu.minAbsoluteSumDiff(
            nums1 = [1,10,4,4,2,7], nums2 = [9,3,5,1,7,4]
        )
    )