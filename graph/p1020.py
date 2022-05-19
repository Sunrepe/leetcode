from typing import List


class Solution:
    def numEnclaves(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j]==1:
                    grid[i][j] = 0
                    a = [(i, j)]
                    cnt, flag = 0, True
                    while a:
                        x, y = a.pop()
                        cnt += 1
                        if x==0 or x == m-1 or y==0 or y==n-1:
                            flag = False
                        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
                            if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == 1:
                                grid[ii][jj] = 0
                                a.append((ii, jj))
                    if flag:
                        ans += cnt
        return ans
    

if __name__ == '__main__':
    slu = Solution()
    print(
        slu.numEnclaves(
            grid = [[0,1,1,0],[0,0,1,0],[0,0,1,0],[0,0,0,0]]
        )
    )