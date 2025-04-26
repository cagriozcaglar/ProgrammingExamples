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

import collections

# Time: O(n)
# Space: O(h)
class Solution:
    # EPI solution
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        Status = collections.namedtuple('Status', ('num_nodes', 'ancestor'))

        def lca_helper(tree, node0, node1):
            if not tree:
                return Status(0, None)

            # Left subtree
            left_result = lca_helper(tree.left, node0, node1)
            # If both nodes found in left subtree, return
            if left_result.num_nodes == 2:
                return left_result

            # Right subtree
            right_result = lca_helper(tree.right, node0, node1)
            # If both nodes found in right subtree, return
            if right_result.num_nodes == 2:
                return right_result

            num_nodes = left_result.num_nodes + \
                        right_result.num_nodes + \
                        int(tree is node0) + \
                        int(tree is node1)
            return Status(num_nodes, tree if num_nodes == 2 else None)

        # Call LCA helper on root, p, q
        return lca_helper(root, p, q).ancestor

    # Leetcode solution
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        ans = 0
        def recurseTree(node: 'TreeNode') -> bool:
            nonlocal ans
            # Reached end of branch, return false
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
