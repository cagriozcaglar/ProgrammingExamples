'''
EPI 9.5: Sum the root-to-leaf paths in a binary tree

Consider a binary tree in which each node contains a binary digit. A root-to-leaf path can be
associated with a binary number-the MSB is at the root, As an example, the binary tree in
Figure 9.4 represents the numbers (1000)2, (1001)2, (10110)2, (110011)2, (11000)2, and (1100)2.

Design an algorithm to compute the sum of the binary numbers represented by the root-to-leaf
paths.
'''
from typing import List

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Time complexity: O(n)
# Space complexity: O(h)
def sum_root_to_leaf(root: TreeNode) -> int:
    def sum_root_to_leaf_helper(root: TreeNode, partial_path_sum: int = 0) -> int:
        if not root:
            return 0

        # Node is not null, so, calculate path sum
        partial_path_sum = partial_path_sum * 2 + root.val

        # If leaf, return the path sum
        if not root.left and not root.right:
            return partial_path_sum

        # If not leaf, return the sum of left and right subtree
        return (sum_root_to_leaf_helper(root.left, partial_path_sum) +
                sum_root_to_leaf_helper(root.right, partial_path_sum))
    
    return sum_root_to_leaf_helper(root)