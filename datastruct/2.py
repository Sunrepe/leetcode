# 单向链表的应用

from typing import List, Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution_mine:
    def glen(self, l:ListNode):
        res = []
        while l.next:
            res.append(l.val)
            l = l.next
        res.append(l.val)
        return res

    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        res, cnt = [], 0
        l1 = self.glen(l1)
        l2 = self.glen(l2)
        if len(l1) > len(l2):
            l1, l2 = l2, l1
        len1, len2 = len(l1), len(l2)
        for i in range(len1):
            val = l1[i]+l2[i]+cnt
            val, cnt = val % 10, val // 10
            res.append(val)
        for i in range(len1, len2):
            val = l2[i]+cnt
            val, cnt = val % 10, val // 10
            res.append(val)
        if cnt: res.append(cnt)
        head = ListNode(res[0])
        current = head
        for i in range(1, len(res)):
            current.next = ListNode(res[i])
            current = current.next
        return head




if __name__ == '__main__':

    slu = Solution()
    print(
        slu.addTwoNumbers(
            l1 = [2,4,3], l2 = [5,6,4]
        )
    )