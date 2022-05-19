from typing import List

class TreeNode:
    def __init__(self, val='0', left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.partent = None

    def __str__(self):
        return str(self.val)


if __name__ == '__main__':

    n, tree_l = input().split()
    n, tree_l = int(n), int(tree_l)

    tmp = input().split()
    inorder = [t for t in tmp]
    tmp = input().split()
    preorder = [t for t in tmp]
    idx_map = {element: i for i, element in enumerate(inorder)}

    def my_tree(in_l, in_r):
        if in_l > in_r:
            return None
        val = preorder.pop(0)  # 最左边的一定是当前子树的根，刚好符合二叉树dfs遍历过程，后序遍历时则只需要pop最后一个元素即可！
        root_index = idx_map[val]
        root = TreeNode(val)
        root.left = my_tree(in_l, root_index - 1)
        root.right = my_tree(root_index + 1, in_r)
        return root
    rot = my_tree(0, len(inorder)-1)

    d = {}  # 记录所有元素的祖先
    def get_ancester(root: TreeNode, ans:List[str]):
        if not root:
            return None
        # 自己是自己的第一个祖先
        ans.append(root.val)
        res = [i for i in ans]
        d[root.val] = res

        if root.left:
            get_ancester(root.left, ans)
        if root.right:
            get_ancester(root.right, ans)
        ans.pop()
    get_ancester(rot, [])

    all_str = set(inorder)

    for _i_ in range(n):
        a, b = input().split()
        flag = []
        if a not in all_str:
            flag.append(a)
        if b not in all_str:
            flag.append(b)

        if len(flag) == 1:
            print('ERROR: {} is not found.'.format(flag[0]))
            continue
        elif len(flag) == 2:
            print('ERROR: {} and {} are not found.'.format(flag[0], flag[1]))
            continue

        da = d[a]
        db = d[b]
        na, nb = len(da), len(db)
        less = min(na, nb)

        ans = 0
        while ans < less and da[ans] == db[ans]:
            ans += 1
        ans -= 1
        if da[ans] == da[-1]:
            print('{} is an ancestor of {}.'.format(a, b))
        elif db[ans] == db[-1]:
            print('{} is an ancestor of {}.'.format(b, a))
        else:
            print('LCA of {} and {} is {}.'.format(a, b, da[ans]))



