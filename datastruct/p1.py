from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        cmp = set()
        for i in range(len(nums)):
            if target-nums[i] in cmp:
                return [nums.index(target-nums[i]), i]
            cmp.add(nums[i])


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.twoSum(
            nums = [3, 3], target = 6
        )
    )