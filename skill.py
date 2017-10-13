
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
    
    
    
subsets: DFS
def subsets(num):
  num.sort()
  subsetshelper([], num, 0)
  
def subsetshelper(path, num, pos):
  #outputsomething
  for i in range(pos, len(num)):
    #unique subsets if i != pos and num[i] == num[i-1]:continue
    path.append something
    subsetshelper(path, num, pos + 1)


permutations
  if length == ..:
    add
 for i in range():
    if list.contain(num[i]):  #visited[i] == True 
      continue


quadratic function [-b/2a,（4ac-b²）/4a]
2d to 1d: (x, y) -> (x*m + y)    x = 1d/m   y = 1d%m
#并查集 （1） 判断是不是一个集合 （2） 合并 （3） 判断环
#trie (1) 一个一个字符遍历 （2） 节省空间 （3）regular expression
'''

