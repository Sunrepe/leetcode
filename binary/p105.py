from typing import List


# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.val)

def print_tree(root: TreeNode):
    A = []
    result = []
    if not root:
        return result
    A.append(root)
    while A:
        current_root = A.pop(0)
        result.append(current_root.val)
        if current_root.left:
            A.append(current_root.left)
        if current_root.right:
            A.append(current_root.right)
    print(result)
    return result


# 最基本的递归实现，增加了索引优化
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        n = len(inorder)
        index = {element: i for i, element in enumerate(inorder)}

        def my_tree(pre, ino):
            if pre[0] > pre[-1]:
                return None
            ino_root = index[preorder[pre[0]]]
            size_left_tree = ino_root - ino[0]      # 求出左边树长度很有用！ 根据树长度进行左右子树构建，因为同一颗子树的前序和中序遍历长度一定是一致的

            root = TreeNode(preorder[pre[0]])
            root.left = my_tree([pre[0]+1, pre[0]+size_left_tree], [ino[0], ino_root-1])
            root.right = my_tree([pre[0]+size_left_tree+1, pre[-1]], [ino_root+1, ino[-1]])
            return root

        return my_tree([0, n-1], [0, n-1])


# 一个简洁的写法，但是该方法会出现两个数组的 的频繁拷贝，时间、空间复杂度都更大
class Solution2:
    def buildTree(self, preorder: List[int], inorder: List[int]):
        if len(inorder) == 0:
            return None
        root = TreeNode()
        root.val = preorder.pop(0)
        ind = inorder.index(root.val)       # 不能写全局索引，因为跟终止条件有关！不推荐
        root.left = self.buildTree(preorder, inorder[:ind])
        root.right = self.buildTree(preorder, inorder[ind+1:])
        return root

# 执行效率高，理解难度高，以后再看
class Solution3:
    def buildTree(self, preorder, inorder):
        def build(stop):
            if inorder and inorder[-1] != stop:
                root = TreeNode(preorder.pop())
                root.left = build(root.val)
                inorder.pop()
                root.right = build(stop)
                return root
        preorder.reverse()
        inorder.reverse()
        return build(None)


if __name__ == '__main__':
    slu = Solution()
    print_tree(
        slu.buildTree(
            preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
        )
    )