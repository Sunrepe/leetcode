from typing import List
import numpy as np
MAXN = 2e9

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        res = set()
        for num in nums:
            if num in res:
                return True
            res.add(num)
        return False


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.containsDuplicate(
            nums = [1,2,3,4]
        )
    )