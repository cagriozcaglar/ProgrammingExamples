"""
Note: This is Leetcode question 559: https://leetcode.com/problems/maximum-depth-of-n-ary-tree/

Given a n-ary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the
null value (See examples).

Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: 3

Example 2:
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: 5

Constraints:
1) The depth of the n-ary tree is less than or equal to 1000.
2) The total number of nodes is between [0, 10^4].
"""

from collections import deque
from typing import Tuple


class Node:
    """
    Definition for a Node.
    """
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class MaximumDepthOfNaryTree:
    def maxDepthRecursive(self, root: 'Node') -> int:
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
        if not root:
            return 0
        else:
            if root.children:
                return 1 + max([self.maxDepthRecursive(child) for child in root.children])
            else:
                return 1

    def maxDepthIterative(self, root: 'Node') -> int:
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
        stack: deque[Tuple[int, Node]] = deque([])
        if root:
            stack.append((1, root))

        depth = 0
        while len(stack) != 0:
            currentDepth, node = stack.pop()
            if node:
                depth = max(depth, currentDepth)
                # NOTE: Check if node.children is null, before iterating over it in a for loop
                if node.children:
                    for child in node.children:
                        stack.append((currentDepth+1, child))
        return depth


if __name__ == "__main__":
    # Object
    maximumDepthOfNaryTree: MaximumDepthOfNaryTree = MaximumDepthOfNaryTree()

    """
    Example 1:
    Input: root = [1,null,3,2,4,null,5,6]
    Output: 3
    
         1
       / |  \
      3  2   4
     / \
    5   6
    
    """
    # Nodes
    node1: Node = Node(1)
    node3: Node = Node(3)
    node2: Node = Node(2)
    node4: Node = Node(4)
    node5: Node = Node(5)
    node6: Node = Node(6)
    # Edges
    node1.children = [node3, node2, node4]
    node3.children = [node5, node6]
    # Test
    print(f"Maximum depth of n-ary tree rooted at {node1.val} using recursive solution: "
          f"{maximumDepthOfNaryTree.maxDepthRecursive(node1)}")
    print(f"Maximum depth of n-ary tree rooted at {node1.val} using iterative solution: "
          f"{maximumDepthOfNaryTree.maxDepthIterative(node1)}")

    """
    Example 2:
    Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
    Output: 5
    """