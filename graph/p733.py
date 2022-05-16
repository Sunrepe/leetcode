from typing import List
from collections import deque

class Solution1:
    # 通过dfs实现
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        # dfs
        old = image[sr][sc]
        if newColor == old:  # 防止陷入死循环
            return image
        def dfs(sr, sc):
            image[sr][sc] = newColor
            for i, j in zip((sr,sr-1,sr,sr+1), (sc-1,sc,sc+1,sc)):
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old:
                    dfs(i, j)
        dfs(sr, sc)
        return image

class Solution2:
    # dfs的最短写法， 一个值得关注的是，通过self调用，image参数变成了全局参数
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc] != newColor:
            old, image[sr][sc] = image[sr][sc], newColor
            for i, j in zip((sr, sr+1, sr, sr-1), (sc+1, sc, sc-1, sc)):
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old:
                    self.floodFill(image, i, j, newColor)
        return image


class Solution:
    # 利用栈或队列实现
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        old = image[sr][sc]
        if newColor == old:  # 防止陷入死循环
            return image
        res = [(sr, sc)]
        image[sr][sc] = newColor
        while res:
            l, r = res.pop()
            for i, j in [(l, r-1),(l-1, r),(l, r+1), (l+1, r)]:
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old:
                    image[i][j] = newColor
                    res.append((i,j))
        return image



if __name__ == '__main__':

    slu = Solution()
    print(
        slu.floodFill(
            image = [[1,1,1],[1,0,0],[1,1,1]], sr = 1, sc = 1, newColor = 2
        )
    )
    a = [2,3]
    a.append(4)
    a.pop()
    a.insert(0,10)