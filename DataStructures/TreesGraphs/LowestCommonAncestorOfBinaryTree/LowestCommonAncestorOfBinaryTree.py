"""
236. Lowest Common Ancestor of a Binary Tree
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.
According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as
the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ans = 0
        def recurseTree(node: 'TreeNode') -> bool:
            nonlocal ans
            # Reached enfd of branch, return false
            if not node:
                return False
            # Left child
            left = recurseTree(node.left)
            # Right child
            right = recurseTree(node.right)
            # if current node is p or q, set mid to True
            mid = (node==p) or (node==q)
            # If any 2 of 3 flags left, right, mid are True, update ans
            if mid+left+right >= 2:
                ans = node
            # Return True if either of three bool vals is True
            return mid or left or right
        recurseTree(root)
        return ans