from typing import List

class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # dp问题，opt[i]表示以i结尾的子数组和的最大值
        opt = [-10**4]
        for num in nums:
            opt.append(max(num, num+opt[-1]))
        return max(opt)


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.maxSubArray(
            nums = [5,4,-1,7,8]
        )
    )