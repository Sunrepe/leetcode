from typing import List


class Solution:
    def countSubIslands(self, grid1: List[List[int]], grid2: List[List[int]]) -> int:
        m, n = len(grid1), len(grid1[0])
        ans = 0
        for i in range(m):
            for j in range(n):
                if grid2[i][j] == 1:
                    a = [(i, j)]
                    grid2[i][j] = 0
                    flag = True
                    while a:
                        x, y = a.pop()
                        if grid1[x][y] != 1:
                            flag = False
                        for ii, jj in zip((x, x - 1, x, x + 1), (y - 1, y, y + 1, y)):
                            if 0 <= ii < m and 0 <= jj < n and grid2[ii][jj] == 1:
                                grid2[ii][jj] = 0
                                a.append((ii, jj))
                    if flag:
                        ans += 1
        return ans


if __name__ == '__main__':
    slu = Solution()
    print(
        slu.countSubIslands(
            grid1=[[1, 0, 1, 0, 1], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 1]],
            grid2=[[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]]
        )
    )