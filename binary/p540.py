from typing import List


class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        n = len(nums)
        l, r = 0, n-1
        while l < r:
            mid = l+r >> 1
            if nums[mid]==nums[mid^1]:
                l = mid+1
            else:
                r = mid
        return nums[l]


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.singleNonDuplicate(
            nums = [1,1,2,3,3,4,4,5,5]
        )
    )