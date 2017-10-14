'''
Union-find / Disjoint-set
If no heuristic, then O(n) for find and union
optimized by path compression and rank
'''

class Node:
    def __init__(self, x, rank = 0):
        self.val = x
        self.rank = rank
        self.parent = self

    def find(self, x):
        if x.parent == x:
            return x
        self.parent = self.find(self.parent)
        return self.parent

    def union(self, y):
        xroot = self.find(self)
        yroot = self.find(y)
        if xroot.rank < yroot.rank:
            xroot.parent = yroot
        elif xroot.rank > yroot.rank:
            yroot.parent = xroot
        elif xroot != yroot:
            yroot.parent = xroot
            xroot.rank += 1