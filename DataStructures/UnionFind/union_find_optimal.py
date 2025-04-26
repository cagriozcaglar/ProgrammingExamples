# UnionFind class Optimized, with Path Compression and Union by Rank
# From https://leetcode.com/explore/learn/card/graph/618/disjoint-set/3843/

# Time complexity:
# Find: O(1) ( O(alpha(n)) to be exact)
# Union: O(1) ( O(alpha(n)) to be exact)
# Connected: O(1) ( O(alpha(n)) to be exact)
class UnionFind:
    def __init__(self, size):
        # Root of each node is itself at the beginning
        self.root = [i for i in range(size)]
        # Use a rank array to record the height of each vertex, i.e., the "rank" of each vertex.
        # The initial "rank" of each vertex is 1, because each of them is
        # a standalone vertex with no connection to other vertices.
        self.rank = [1] * size

    # The find function here is the same as that in the disjoint set with ** path compression **.
    def find(self, x):
        # If the vertex is the root, its parent is itself
        if x == self.root[x]:  
            return x
    	# Recursion. Some ranks may become obsolete so they are not updated
        self.root[x] = self.find(self.root[x])
        return self.root[x]

    # The union function with ** union by rank **
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.root[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.root[rootX] = rootY
            else:
                self.root[rootY] = rootX
                self.rank[rootX] += 1  # Only time rank of a node changes in union() method

    def connected(self, x, y):
        return self.find(x) == self.find(y)


# Test Case
uf = UnionFind(10)
# 1-2-5-6-7 3-8-9 4
uf.union(1, 2)
uf.union(2, 5)
uf.union(5, 6)
uf.union(6, 7)
uf.union(3, 8)
uf.union(8, 9)
print(uf.connected(1, 5))  # true
print(uf.connected(5, 7))  # true
print(uf.connected(4, 9))  # false
# 1-2-5-6-7 3-8-9-4
uf.union(9, 4)
print(uf.connected(4, 9))  # true
