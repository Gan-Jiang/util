'''
binary indexed tree. Used for random range sum.
Query O(log n), update O(log n)
'''

class NumArray(object):
    def __init__(self, nums):
        self.size = len(nums)
        self.nums = nums
        self.tree = self.build_tree()


    def build_tree(self):
        '''
        initial the binary indexed tree.
        :return:
        '''
        tree = [0] * (self.size + 1)
        for i in range(self.size):
            k = i + 1
            while k <= self.size:
                tree[k] += self.nums[i]
                k += (k & -k)
        return tree


    def update(self, i, val):
        delta = val - self.nums[i]
        self.nums[i] = val
        i += 1
        while i <= self.size:
            self.tree[i] += delta
            i += (i & -i)


    def prefix_sum(self, i):
        res_sum = 0
        i += 1
        while i > 0:
            res_sum += self.tree[i]
            i -= (i & -i)
        return res_sum


    def sumRange(self, i, j):
        return self.prefix_sum(j) - self.prefix_sum(i - 1)