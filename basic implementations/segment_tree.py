"""
    Segment Tree for range sum queries. Have not implemented lazy propagation.
    Creating the tree takes O(n) time. Query and updates are both O(log n).

"""


# Segment tree node
class Node(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.total = 0
        self.left = None
        self.right = None


class NumArray(object):
    def __init__(self, nums):
        """
        initialize your data structure here.
        :type nums: List[int]
        """
        self.root = self.createTree(nums, 0, len(nums) - 1)


    def createTree(self, nums, l, r):
        # helper function to create the tree from input array
        # base case
        if l > r:
            return None

        # leaf node
        if l == r:
            n = Node(l, r)
            n.total = nums[l]
            return n

        mid = (l + r) // 2

        root = Node(l, r)

        # recursively build the Segment tree
        root.left = self.createTree(nums, l, mid)
        root.right = self.createTree(nums, mid + 1, r)
        # Total stores the sum of all leaves under root
        root.total = root.left.total + root.right.total

        return root


    def _sumRange(self, root, i, j):
        # Helper function to calculate range sum
        # If the range exactly matches the root, we already have the sum
        if root.start >= i and root.end <= j:
            return root.total

        mid = (root.start + root.end) // 2

        # If end of the range is less than the mid, the entire interval lies
        # in the left subtree
        if j <= mid:
            return self._sumRange(root.left, i, j)

        # If start of the interval is greater than mid, the entire inteval lies
        # in the right subtree
        elif i >= mid + 1:
            return self._sumRange(root.right, i, j)

        # Otherwise, the interval is split. So we calculate the sum recursively,
        # by splitting the interval
        else:
            return self._sumRange(root.left, i, mid) + self._sumRange(root.right, mid + 1, j)


    def sumRange(self, i, j):
        """
        sum of elements nums[i..j], inclusive.
        :type i: int
        :type j: int
        :rtype: int
        """
        if i > self.root.end or j < self.root.start:  # if the range is not overlapped with our root range.
            return 0
        if i < self.root.start:
            i = self.root.start
        if j > self.root.end:
            j = self.root.end
        return self._sumRange(self.root, i, j)


    def _updateVal(self, root, i, val):
        # Helper function to update a value
        # Base case. The actual value will be updated in a leaf.
        # The total is then propogated upwards
        if root.start == root.end:
            root.total = val
            return val

        mid = (root.start + root.end) // 2

        # If the index is less than the mid, that leaf must be in the left subtree
        if i <= mid:
            self._updateVal(root.left, i, val)

        # Otherwise, the right subtree
        else:
            self._updateVal(root.right, i, val)

        # Propogate the changes after recursive call returns
        root.total = root.left.total + root.right.total

        return root.total


    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: int
        """
        if self.root.start <= i <= self.root.end:
            self._updateVal(self.root, i, val)