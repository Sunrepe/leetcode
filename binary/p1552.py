from typing import List

class Solution:
    def maxDistance(self, position: List[int], m: int) -> int:
        position.sort()
        # 检验当前x：以x为上限切分数组，数组需要切成几段？
        def check(x):
            cnt, minx = 1, min(position)
            for p in position:
                if p >= minx+x:
                    cnt += 1
                    minx = p
            return True if cnt >= m else False
        # 搜索最大子序列和x 使得可以将原数组拆成m份
        # 利用红蓝树实现
        l, r = 0, max(position)+1
        while l+1 < r:
            mid = l+r >> 1
            if not check(mid):
                r = mid
            else:
                l = mid
        return l


if __name__ == '__main__':

    slu = Solution()
    print(
        slu.maxDistance(
            position = [5,4,3,2,1,1000000000], m = 2
        )
    )