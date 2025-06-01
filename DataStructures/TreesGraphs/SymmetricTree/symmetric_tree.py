'''
Leetcode 101: Symmetric Tree

Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center).

Example 1:
Input: root = [1,2,2,3,4,4,3]
Output: true

Example 2:
Input: root = [1,2,2,null,3,null,3]
Output: false
'''

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from typing import List, Optional

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def is_mirror(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
            if not p or not q:  # If one of them is non-null, return p == q
                return p == q
            return (p.val == q.val) and \
                   is_mirror(p.left, q.right) and \
                   is_mirror(p.right, q.left)

        return is_mirror(root, root)
