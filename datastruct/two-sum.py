from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashtable = dict()
        for i, num in enumerate(nums):
            if target - num in hashtable:
                return [hashtable[target - num], i]
            hashtable[nums[i]] = i
        return []

    def twoSum1(self, nums: List[int], target: int) -> List[int]:
        s = set(nums)
        st, ed, tt = -1, -1, -1
        for i, num in enumerate(nums):
            res = target-num
            st = i
            for i in range(i + 1, len(nums)):
                if nums[i] == res:
                    ed = i
                    return [st, ed]
        return [st, ed]




if __name__ == '__main__':

    slu = Solution()
    print(
        slu.twoSum(
            nums = [3,2,4], target = 6
        )
    )