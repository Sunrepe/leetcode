from typing import List


class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1.sort()
        nums2.sort()
        m, n = len(nums1), len(nums2)
        ans = []
        l, r = 0, 0
        while l < m and r < n:
            if nums1[l] == nums2[r]:
                ans.append(nums1[l])
                l += 1
                r += 1
            elif nums1[l] < nums2[r]:
                l += 1
            else:
                r += 1
        return ans


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.intersect(
            nums1 = [4,9,5], nums2 = [9,4,9,8,4]
        )
    )