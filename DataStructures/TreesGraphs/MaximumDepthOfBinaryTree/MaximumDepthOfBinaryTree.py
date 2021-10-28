"""
Note: This is Leetcode question 104: https://leetcode.com/problems/maximum-depth-of-binary-tree/

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf
node.

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 3

Example 2:
Input: root = [1,null,2]
Output: 2

Example 3:
Input: root = []
Output: 0

Example 4:
Input: root = [0]
Output: 1

Constraints:
1) The number of nodes in the tree is in the range [0, 104].
2) 100 <= Node.val <= 100
"""

from collections import deque
from typing import Tuple


class TreeNode:
    """
    Definition for a binary tree node.
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class MaximumDepthOfBinaryTree:
    @staticmethod
    def maxDepthRecursive(root: TreeNode) -> int:
        """
        Recursion + DFS (It could as well be DFS)
         - Run-time complexity: We visit each node exactly once, thus the time complexity is O(N), where N is the number
           of nodes.
         - Space complexity:
           - Worst case: The tree is completely unbalanced, e.g. each node has only left child node, the recursion call
             would occur N times (the height of the tree), therefore the storage to keep the call stack would be O(N).
           - Best case: The tree is completely balanced, the height of the tree would be log(N). Therefore, the space
             complexity in this case would be O(log(N).
        Note: One might have noticed that the above recursion solution is probably not the most optimal one in terms of
        the space complexity, and in the extreme case the overhead of call stack might even lead to stack overflow. To
        address the issue, one can tweak the solution a bit to make it tail recursion, which is a specific form of
        recursion where the recursive call is the last action in the function.
        :param root:
        :return:
        """
        return 0 if not root else 1 + max(MaximumDepthOfBinaryTree.maxDepthRecursive(root.left),
                                          MaximumDepthOfBinaryTree.maxDepthRecursive(root.right))

    @staticmethod
    def maxDepthIterative(root: TreeNode) -> int:
        """
        Iterative solution using stack (to mimic recursive solution)
        We could also convert the above recursion into iteration, with the help of the stack data structure. Similar
        to the behaviors of the function call stack, the stack data structure follows the pattern of LIFO
        (Last-In-First-Out), i.e. the last element that is added to a stack would come out first. With the help of the
        stack data structure, one could mimic the behaviors of function call stack that is involved in the recursion,
        to convert a recursive function to a function with iteration.
        We start from a stack which contains the root node and the corresponding depth which is 1. Then we proceed to
        the iterations: pop the current node out of the stack and push the child nodes. The depth is updated at each step.
         - Time complexity: O(N).
         - Space complexity:
           - In the worst case, the tree is completely unbalanced, e.g. each node has only left child node, the recursion
             call would occur NN times (the height of the tree), therefore the storage to keep the call stack would be O(N).
           - In the average case (the tree is balanced), the height of the tree would be log(N). Therefore, the space
             complexity in this case would be O(log(N)).
        :param root:
        :return:
        """
        stack: deque[Tuple[int, TreeNode]] = deque([])
        if root:
            stack.append((1, root))

        depth = 0
        while len(stack) != 0:
            currentDepth, node = stack.pop()
            if node:
                depth = max(depth, currentDepth)
                stack.append((currentDepth+1, node.left))
                stack.append((currentDepth+1, node.right))
        return depth


if __name__ == "__main__":
    """
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: 3
    
         3
       /  \
      9    20
          / \
        15   7

    """
    # Nodes
    node3: TreeNode = TreeNode(3)
    node9: TreeNode = TreeNode(9)
    node20: TreeNode = TreeNode(20)
    node15: TreeNode = TreeNode(15)
    node7: TreeNode = TreeNode(7)
    # Edges
    node3.left = node9
    node3.right = node20
    node20.left = node15
    node20.right = node7
    # Test
    print(f"Maximum depth of tree rooted at {node3.val} using recursive solution: "
          f"{MaximumDepthOfBinaryTree.maxDepthRecursive(node3)}")
    print(f"Maximum depth of tree rooted at {node3.val} using iterative solution: "
          f"{MaximumDepthOfBinaryTree.maxDepthIterative(node3)}")

    """
    Example 2:
    Input: root = [1,null,2]
    Output: 2
    
    Example 3:
    Input: root = []
    Output: 0
    
    Example 4:
    Input: root = [0]
    Output: 1
    """