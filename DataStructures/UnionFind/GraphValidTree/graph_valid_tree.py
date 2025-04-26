from typing import List

class UnionFind:
    def __init__(self, n):
        self.parent = [node for node in range(n)]  # parents of nodes set to themselves
        self.rank = [1] * n

    # find(), with path compression
    def find(self, A):
        if A == self.parent[A]:  # A is root
            return A
        self.parent[A] = self.find(self.parent[A])
        return self.parent[A]

    # union(), union with rank
    # Returns true if merge happened, false otherwise
    def union(self, A, B):
        root_A = self.find(A)
        root_B = self.find(B)
        # If roots are the same, from same group, no merge needed, return False
        if root_A == root_B:
            return False
        # Merge: ensure the larger set remains root
        if self.rank[root_A] < self.rank[root_B]:
            self.parent[root_A] = root_B
        elif self.rank[root_A] < self.rank[root_B]:
            self.parent[root_B] = root_A
        else:
            self.parent[root_B] = root_A
            self.rank[root_A] += 1
        return True

class Solution:
    # Solution 2: Advanced Graph Theory + DFS
    # 1. Check whether or not there are n - 1 edges. If there's not, then return false.
    # 2. Check whether or not the graph is fully connected. Return true if it is, false otherwise.
    # Time: O(n)
    # Space: O(n)
    def validTreeSolution2(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False

        # Create adjacency matrix
        adj_list = [[] for _ in range(n)]
        for src, dst in edges:
            adj_list[src].append(dst)
            adj_list[dst].append(src)

        # Run DFS, start from first node, check if all nodes are seen at the end
        seen = set()

        def dfs(node):
            if node in seen:
                return
            seen.add(node)
            for neighbour in adj_list[node]:
                dfs(neighbour)

        # Start DFS
        dfs(0)
        # Check if all nodes are seen
        return len(seen) == n
        
    # Solution 3: Advanced Graph Theory + Union-Find
    # Time: O(n * alpha(n))
    # Space: O(n * alpha(n))
    def validTreeWithUnionFind(self, n: int, edges: List[List[int]]) -> bool:
        if len(edges) != n - 1:
            return False

        union_find = UnionFind(n)

        # Add each edge. Check if merge happened, if it didn't, there must be a cycle.
        for A, B in edges:
            if not union_find.union(A, B):
                return False

        # No cycles found, return true
        return True