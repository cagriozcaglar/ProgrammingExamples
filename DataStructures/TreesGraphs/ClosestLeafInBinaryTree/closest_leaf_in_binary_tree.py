'''
Leetcode 742. Closest Leaf in a Binary Tree

Given the root of a binary tree where every node has a unique value and a target integer k, return the value of the nearest leaf node to the target k in the tree.

Nearest to a leaf means the least number of edges traveled on the binary tree to reach any leaf of the tree. Also, a node is called a leaf if it has no children.
'''
import  collections
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Time : O(N)
# Space: O(N)
class Solution:
    def findClosestLeaf(self, root: Optional[TreeNode], k: int) -> int:
        # 1. Construct graph from BT
        graph = collections.defaultdict(list)  # adjacency list

        # 1.1. DFS
        def dfs(node, parent=None):
            if node:
                graph[node].append(parent)
                graph[parent].append(node)
                dfs(node.left, node)
                dfs(node.right, node)

        # 1.2. Populate graph, starting from root
        dfs(root)

        # 2. Run BFS to find the nearest leaf node to target k in the tree
        # 2.1. Push all nodes with value k to BFS queue
        queue = collections.deque(node for node in graph if node and node.val == k)
        # 2.2. Set seen node set
        seen = set(queue)

        # 2.3. Iterate on BFS queue until empty
        while queue:
            node = queue.popleft()
            if node:
                # Found leaf node: node only has parent (num=1), or no parent (num=0)
                if len(graph[node]) <= 1:
                    return node.val
                # If not lead node, go through node's neighbours
                for neighbour in graph[node]:
                    # If neighbour not seen, add to queue, update seen
                    if neighbour not in seen:
                        seen.add(neighbour)
                        queue.append(neighbour)
