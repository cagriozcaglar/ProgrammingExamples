"""
270. Closest Binary Search Tree Value
Given the root of a binary search tree and a target value, return the value in the BST that is closest to the target.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def closestValue(self, root: TreeNode, target: float) -> int:
        """
        Time complexity: O(H)
        """
        closest = root.val
        while root:
            if abs(root.val-target) < abs(closest-target):
                closest = root.val
            root = root.left if target < root.val else root.right
        return closest