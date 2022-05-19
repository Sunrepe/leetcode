from typing import List


class Solution:
    def findClosestElements(self, arr, k, x):
        left = 0
        right = len(arr) - k - 1
        while (left <= right) :
            mid = left+right >> 1
            if (x - arr[mid] > arr[mid + k] - x) :
                left = mid + 1
            else :
                right = mid - 1
        return arr[left : left + k]


class Solution1:
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        length = len(arr)

        # 二分找到最接近x的数
        l, r = 0, len(arr) - 1
        while (l < r):
            mid = int((l + r) / 2)
            if (abs(x - arr[mid]) > abs(x - arr[mid + 1]) or (arr[mid]==arr[mid+1] and arr[mid] < x) ):
                l = mid + 1
            else:
                r = mid

        # 双指针从中间向两边遍历
        a,b = r,r
        k -= 1
        while(k>0):
            if a == 0:
                b += 1
                k -= 1
            elif b == length-1:
                a -= 1
                k -= 1

            elif(abs(x-arr[a-1]) <= abs(x-arr[b+1])):
                a -= 1
                k -= 1

            else:
                b += 1
                k -= 1
        return arr[a:b+1]


class eSolution:
    def triangleNumber(self, nums: List[int]) -> int:
        length = len(nums)


if __name__ == '__main__':

    slu = Solution()
    print(slu.findClosestElements(arr = [1,2,2,2,2,2,2,3], k = 3, x = 3))