from typing import List


class Solution:
    def maxFrequency(self, nums: List[int], k: int) -> int:
        nums.sort()
        n = len(nums)
        tot, ans = 0, 1
        l, r = 0, 1
        while r<n:
            tot += (nums[r]-nums[r-1]) * (r-l)   # 按行增加面积
            while tot > k:
                tot -= (nums[r]-nums[l])    # 按列介绍面积
                l += 1
            ans = max(ans, r-l+1)
            r += 1
        return ans



if __name__ == '__main__':
    slu = Solution()
    print(
        slu.maxFrequency(
            nums = [1,4,2,3,4,8,13], k = 5
        )
    )