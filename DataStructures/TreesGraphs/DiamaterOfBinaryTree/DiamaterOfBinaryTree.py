"""
543. Diameter of Binary Tree
Given the root of a binary tree, return the length of the diameter of the tree.
The diameter of a binary tree is the length of the longest path between any two nodes in a tree.
This path may or may not pass through the root.
The length of a path between two nodes is represented by the number of edges between them.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        diameter = 0

        def longestPath(node) -> int:
            nonlocal diameter
            if not node:
                return 0
            # Find longest path rooted at left and right children
            leftLongestPath = longestPath(node.left)
            rightLongestPath = longestPath(node.right)
            # Update diameter
            diameter = max(diameter, leftLongestPath+rightLongestPath)
            # Return the longer of left and right longest path, +1 for the parent
            return max(leftLongestPath, rightLongestPath) + 1

        longestPath(root)
        return diameter