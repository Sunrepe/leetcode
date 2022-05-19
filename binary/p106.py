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
    def buildTree(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        
        n = len(inorder)
        idx_map = {element: i for i, element in enumerate(inorder)}

        def my_tree(in_left, in_right):
            # 终止条件判断
            if in_left > in_right:
                return None
            val = postorder.pop()        # 后续遍历，先构造右子树

            # 后续遍历
            root_index = idx_map[val]
            root = TreeNode(val)
            root.right = my_tree(root_index+1, in_right)
            root.left = my_tree(in_left, root_index-1)
            return root

        return my_tree(0, n-1)


if __name__ == '__main__':
    slu = Solution()
    print_tree(
        slu.buildTree(
            inorder = [9,3,15,20,7], postorder = [9,15,7,20,3]
        )
    )