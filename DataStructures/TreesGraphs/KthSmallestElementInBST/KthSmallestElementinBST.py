"""
230. Kth Smallest Element in a BST
Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        def inorder(node):
            return inorder(node.left) + [node.val] + inorder(node.right) if node else []

        return inorder(root)[k-1]

    def kthSmallestIterative(self, root: TreeNode, k: int) -> int:
        stack = []

        while True:
            # Go to leftmost node
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            k -= 1
            if not k:
                return root.val
            root = root.right
