
#arithmetic sequence sn = n*(a1 + an) / 2   an = a1 + (n-1) * d
#geometric sequence an = a1*q^{n-1}   sn = a1*(1-q^n) / (1-q)

#binary search:  while start + 1 < end:
                    # mid = start + (end - start) // 2

'''
DFS:
traverse
def traverse(root):
    if not root:
        return

    traverse(root.left)
    traverse(root.right)


divide & conquer
def traverse(root)
    if not root:
        return something

    left = traverse(root.left)
    right = traverse(root.right)
    result = merge from left and right
    return result



subsets:
def subsets(self, nums):
    """
    :type nums: List[int]
    :rtype: List[List[int]]
    """
    res = []
    self.dfs(nums, 0, [], res)
    return res


def dfs(self, nums, index, path, res):
    res.append(path)
    for i in range(index, len(nums)):
        self.dfs(nums, i+1, path+[nums[i]], res)



BFS: 2 queues; 1queue + dummy node; 1queue

def BFS(root):
    if not root:
        return something
    queue.append(root)
    while queue:
        size = len(queue)
        for i in range(size):
            x = queue.pop(0)
            something
            if x.left:
                queue.append(x.left)
            if x.right:
                queue.append(x.right)
    return something.
'''

class TreeNode(object):
     def __init__(self, x):
         self.val = x
         self.left = None
         self.right = None

def flatten(root):
    """
    :type root: TreeNode
    :rtype: void Do not return anything, modify root in-place instead.
    """
    if not root:
        return
    if root.val == 5 or root.val == 3:
        aaa=1
    if root.left:
        flatten(root.left)

    if root.right:
        flatten(root.right)
    left = root.left
    right = root.right
    root.left = None
    if left:
        root.right = left
        if right:
            root.right.right = right
    else:
        if right:
            root.right = right

root = TreeNode(1)
root1 = TreeNode(2)
root2 = TreeNode(3)
root3 = TreeNode(4)
root4 = TreeNode(5)
root.left = root1
root1.left = root2
root1.right = root3
root2.left = root4
flatten(root)