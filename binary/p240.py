from typing import List


class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        l, r = 0, n-1
        while l<m and r>-1:
            if matrix[l][r] == target:
                return True
            elif matrix[l][r] < target:
                l += 1
            else:
                r -= 1
        return False


# 按对角线检索定位到关键行和列
class Solution1:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m, n = len(matrix), len(matrix[0])
        # 按行
        if m <= n:
            for l1 in range(m):
                l, r = -1, n
                while l + 1 < r:
                    mid = l + r >> 1
                    if matrix[l1][mid] == target:
                        return True
                    elif matrix[l1][mid] < target:
                        l = mid
                    else:
                        r = mid
        else:
            for r1 in range(n):
                l, r = -1, m
                while l + 1 < r:
                    mid = l + r >> 1
                    if matrix[mid][r1] == target:
                        return True
                    elif matrix[mid][r1] < target:
                        l = mid
                    else:
                        r = mid


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.searchMatrix(
            matrix=[[5,6,10,14],[6,10,13,18],[10,13,18,19]], target=14
        )
    )