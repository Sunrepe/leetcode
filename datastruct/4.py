# 采用类似于2分方法可以实现

from typing import List


class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        tgt = (m+n)>>1
        l1, r1 = -1, m
        l2, r2 = -1, n
        les = 0
        while les != tgt:
            mid1 = (l1+r1)>>1
            mid2 = (l2 + r2) >> 1
            les = mid1+mid2
            if les < tgt:
                if nums1[mid1] <= nums2[mid2]:
                    l1 = mid1
                else:
                    l2 = mid2
            elif les>tgt:
                if nums1[mid1] <= nums2[mid2]:
                    r2 = mid2
                else:
                    r1 = mid1
            print(11)
        print(111)
        if (m+n)%2:
            return min(nums1[mid1], nums2[mid2])
        else:
            pass







if __name__ == '__main__':
    slu = Solution()
    print(
        slu.findMedianSortedArrays(
            nums1=[1,2,3,4,5,6,7,8,9,10], nums2=[1,2,13,14]
        )
    )