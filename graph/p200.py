from typing import List

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # dfs
        old, newColor= "1", "0"
        m, n = len(grid), len(grid[0])
        
        res = 0
        for x in range(m):
            for y in range(n):
                if grid[x][y] == old:
                    res += 1
                    a = [(x, y)]
                    while a:
                        l, r = a.pop()
                        grid[l][r] = newColor
                        for i, j in [(l, r - 1), (l - 1, r), (l, r + 1), (l + 1, r)]:
                            if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == old:
                                a.append((i, j))

        return res


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.numIslands(
            grid=[
                ["1", "1", "0", "0", "0"],
                ["1", "1", "0", "0", "0"],
                ["0", "0", "1", "0", "0"],
                ["0", "0", "0", "1", "1"]
            ]
        )
    )