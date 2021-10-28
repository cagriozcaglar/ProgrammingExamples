"""
Longest Common Ancestor of Deepest Leaves of a Binary Tree
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/discuss/334577/JavaC%2B%2BPython-Two-Recursive-Solution
# https://leetcode.com/problems/lowest-common-ancestor-of-deepest-leaves/discuss/479076/Python-Recursive-and-Iterative-Solution
from typing import List, Optional
class Solution:
    def lcaDeepestLeaves(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        return self.helper(root)[0]

    def helper(self, root):
        if not root:
            return None, 0
        lcaLeft, heightLeft = self.helper(root.left)
        lcaRight, heightRight = self.helper(root.right)

        if heightLeft > heightRight:
            return lcaLeft, heightLeft+1
        elif heightLeft < heightRight:
            return lcaRight, heightRight+1
        else:
            return root, heightLeft+1