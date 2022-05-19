from typing import List


class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        minx = prices[0]
        for i in range(len(prices)):
            minx, prices[i] = min(minx, prices[i]), prices[i]-minx
        return max(prices)


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.maxProfit(
            prices = [7,1,5,3,6,4]
        )
    )