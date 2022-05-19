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


class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        n = len(preorder)
        index = {element: i for i, element in enumerate(inorder)}

        def my_tree(pre, ino):
            if pre[0] > pre[-1]:
                return None
            ino_root = index[preorder[pre[0]]]
            size_left_tree = ino_root - ino[0]      # 求出左边树长度很有用！ 根据树长度进行左右子树构建

            root = TreeNode(preorder[pre[0]])

            root.left = my_tree([pre[0]+1, pre[0]+size_left_tree], [ino[0], ino_root-1])
            root.right = my_tree([pre[0]+size_left_tree+1, pre[-1]], [ino_root+1, ino[-1]])

            return root

        return my_tree([0, n-1], [0, n-1])





if __name__ == '__main__':
    slu = Solution()
    print_tree(
        slu.buildTree(
            preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
        )
    )