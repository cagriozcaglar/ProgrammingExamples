"""
863. All Nodes Distance K in Binary Tree
Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of
all nodes that have a distance k from the target node.
You can return the answer in any order.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
        self.parent = None

class Solution:
    def distanceK(self, root: TreeNode, target: TreeNode, k: int) -> List[int]:
        # Add parent pointers to all nodes using DFS
        def dfs(node, parent=None):
            if node:
                node.parent = parent
                dfs(node.left, node)
                dfs(node.right, node)

        # Call DFS on root and add parent pointers to all nodes
        dfs(root)

        # Do BFS from target node, and find nodes distance-k
        bfsQueue = collections.deque( [(target, 0)] )
        visited = {target}
        while bfsQueue:
            # Found k-distance nodes, return
            if bfsQueue[0][1] == k:
                return [node.val for node, distance in bfsQueue]
            node, distance = bfsQueue.popleft()
            for neigh in (node.left, node.right, node.parent):
                if neigh and neigh not in visited:
                    visited.add(neigh)
                    bfsQueue.append( (neigh, distance+1) )
        return []