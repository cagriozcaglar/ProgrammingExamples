"""
Delete node from BST. Leetcode 450
"""
from typing import Optional

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:

    # Find successor of a node: One step right, then always left
    def successor(self, root):
        root = root.right
        while root.left:
            root = root.left
        return root.val

    # Find predecessor of a node: One step left, then always right
    def predecessor(self, root: Optional[TreeNode]):
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None

        # Delete from right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # Delete from left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        # Node found, delete in 3 cases
        elif root.val == key:
            # Case 1: The node is a leaf
            if not (root.left or root.right):
                # Delete the root node
                root = None
            # Case 2: The node is not a leaf, has right child
            # Replace node with its successor, leftmost node in right subtree
            elif root.right:
                # 2.1. Replace root's value with successor's value
                root.val = self.successor(root)
                # 2.2. Delete successor node
                root.right = self.deleteNode(root.right, root.val)
            # Case 3: The node is not a leaf, has left child
            # Replace node with its predecessor, rightmost node in left subtree
            else:
                # 2.1. Replace root's value with predecessor's value
                root.val = self.predecessor(root)
                # 2.2. Delete successor node
                root.left = self.deleteNode(root.left, root.val)
        return root