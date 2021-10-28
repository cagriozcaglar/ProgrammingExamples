"""
Convert Binary Search Tree to Sorted Doubly Linked List
Recursive Inorder Traversal practice with some tweaks
"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        # Performs inorder traversal: left -> node -> right, create DLL
        def helper(node):
            nonlocal last, first
            if node:
                # 1. L: left
                helper(node.left)
                # 2. C: Current node
                if last:
                    # Link prev node (last) with current node (node)
                    last.right = node
                    node.left = last
                else:
                    # First node, set it
                    first = node
                last = node  # Set last node to be the current node
                # 3. R: right
                helper(node.right)

        if not root:
            return None
        # First / Last (Smallest / Largest) nodes
        first, last = None, None
        helper(root)
        # Close DLL: connect first and last both ways
        last.right = first
        first.left = last
        return first