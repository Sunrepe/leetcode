from typing import List

class Solution:
    def closedIsland(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        cnt = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    a, flag = [(i, j)], True
                    while a:
                        x, y = a.pop()
                        if x==0 or x == m-1 or y==0 or y==n-1:
                            flag = False
                        for ii, jj in zip((x,x-1,x,x+1),(y-1,y,y+1,y)):
                            if 0 <= ii < m and 0 <= jj < n and grid[ii][jj] == 0:
                                grid[ii][jj] = 1
                                a.append((ii,jj))
                    if flag:
                        cnt += 1
        return cnt



if __name__ == '__main__':
    slu = Solution()
    print(
        slu.closedIsland(
            grid = [[1,1,1],[1,0,1],[0,1,0]]
        )
    )