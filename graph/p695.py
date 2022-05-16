import collections
from typing import List


class Solution1:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        maxn = 0
        m, n = len(grid), len(grid[0])
        for i in range(m):
            for j in range(n):
                cnt = 0
                if grid[i][j] == 1:
                    grid[i][j] = 0
                    # a = [(i, j)]
                    a = collections.deque([(i,j)])
                    while a:
                        sr, sc = a.pop()
                        cnt += 1
                        for l, r in zip((sr-1,sr,sr+1,sr),(sc,sc-1,sc,sc+1)):
                            if 0<=l<m and 0<=r<n and grid[l][r]==1:
                                a.append((l, r))
                                grid[l][r] = 0
                    maxn = max(maxn, cnt)
        return maxn


class Solution:
    def bfs(self, grid, x, y, target, new):
        m, n = len(grid), len(grid[0])
        if x<0 or x>=m or y<0 or y>=n or grid[x][y]!=target:
            return 0
        ans = 1
        grid[x][y] = new
        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
            ans += self.bfs(grid, ii, jj, target, new)
        return ans


    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # dfs
        target, new = 1, 0
        m, n = len(grid), len(grid[0])
        maxn = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == target:
                    maxn = max(maxn, self.bfs(grid, i, j, target, new))
        return maxn


if __name__ == '__main__':

    slu = Solution1()
    print(
        slu.maxAreaOfIsland(
            [[1,1,0,0,0],[1,1,0,0,0],[0,0,0,1,1],[0,0,0,1,1]]
        )
    )