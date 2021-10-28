"""
124. Binary Tree Maximum Path Sum
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting
them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
The path sum of a path is the sum of the node's values in the path.
Given the root of a binary tree, return the maximum path sum of any path.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def maxGain(node):
            nonlocal maxSum
            if not node:
                return 0
            # Max sum on left / right subtrees of node
            leftGain = max(maxGain(node.left), 0)
            rightGain = max(maxGain(node.right), 0)
            # Price to start a new path where node is highest
            priceNewPath = node.val + leftGain + rightGain
            # Update maxSum if better to start a new path
            maxSum = max(maxSum, priceNewPath)
            # Return max gain if continue the same path
            return node.val + max(leftGain, rightGain)
        maxSum = float('-inf')
        maxGain(root)
        return maxSum