"""
814. Binary Tree Pruning
Given the root of a binary tree, return the same tree where every subtree (of the given tree) not containing a 1
has been removed.
A subtree of a node node is node plus every node that is a descendant of node.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def containsOne(self, node: Optional[TreeNode]) -> bool:
        # Null check: Do this before calling left / right child later
        if not node:
            return False
        # Does left / right contain 1
        leftContainsOne = self.containsOne(node.left)
        rightContainsOne = self.containsOne(node.right)
        # Prune left / right child is they don't contain 1
        if not leftContainsOne:
            node.left = None
        if not rightContainsOne:
            node.right = None
        # Return whether at least one of node, left / right child has 1
        return node.val == 1 or leftContainsOne or rightContainsOne

    def pruneTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        return root if self.containsOne(root) else None