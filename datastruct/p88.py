from typing import List


class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        l, r = m-1, n-1
        tmp = len(nums1)-1
        while l>=0 and r >= 0:
            if nums1[l]>nums2[r]:
                nums1[tmp] = nums1[l]
                tmp -= 1
                l -= 1
            else:
                nums1[tmp] = nums2[r]
                tmp -= 1
                r -= 1
        while l >= 0:
            nums1[tmp] = nums1[l]
            tmp -= 1
            l -= 1
        while r>= 0:
            nums1[tmp] = nums2[r]
            tmp -= 1
            r -= 1



if __name__ == '__main__':

    slu = Solution()
    print(
        slu.merge(
            nums1=[1, 2, 3, 0, 0, 0], m=3, nums2=[2, 5, 6], n=3
        )
    )


