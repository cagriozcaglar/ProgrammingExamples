"""
Note: This is Leetcode question 111: https://leetcode.com/problems/minimum-depth-of-binary-tree/

Given a binary tree, find its minimum depth.
The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 2

Example 2:
Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5

Constraints:
1) The number of nodes in the tree is in the range [0, 105].
2) -1000 <= Node.val <= 1000
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


class MinimumDepthOfBinaryTree:
    def minDepthRecursive(self, root: TreeNode) -> int:
        """
        Recursive solution.
        Check all cases below, this is different from max depth of binary tree question
        1. Base case: Leaf node. Return 1, because height of node is 1
        2. Left subtree is null (right subtree is non-null)
        3. Right subtree is null (left subtree is non-null)
        4. Left and Right subtree are both non-null

        This method requires traversing all nodes of the tree (although not necessary)

         - Time complexity: We visit each node exactly once, thus the time complexity is O(N), where N is number of nodes.
         - Space complexity: In the worst case, the tree is completely unbalanced, e.g. each node has only one child
           node, the recursion call would occur N times (the height of the tree), therefore the storage to keep the call
           stack would be O(N). But in the best case (the tree is completely balanced), the height of the tree would be
           log(N). Therefore, the space complexity in this case would be O(log(N)).

        Note: See the solution below for the faster one.
        """
        if not root:
            return 0
        # Base case: Leaf node. Return 1, because height of node is 1
        if not root.left and not root.right:
            return 1
        # Left subtree is null (right subtree is non-null, because previous condition is passed))
        if not root.left:
            return 1 + self.minDepthRecursive(root.right)
        # Right subtree is null (left subtree is non-null, because previous condition is passed))
        if not root.right:
            return 1 + self.minDepthRecursive(root.left)
        # Else (both left and right subtree are non-null), return 1 + min(leftSubtreeMinDepth, rightSubtreeMinDepth)
        return 1 + min(self.minDepthRecursive(root.left), self.minDepthRecursive(root.right))

    def minDepthBfsIterative(self, root: TreeNode) -> int:
        """
        Solution with BFS:
        The drawback of the DFS approach (recursive like the one above, or iterative) in this case is that all nodes
        should be visited to ensure that the minimum depth would be found. Therefore, this results in a O(N)
        complexity. One way to optimize the complexity is to use the BFS strategy. We iterate the tree level by level,
        and the first leaf we reach corresponds to the minimum depth. As a result, we do not need to iterate all nodes.

        - Time complexity: In the worst case for a balanced tree we need to visit all nodes level by level up to the
          tree height, that excludes the bottom level only. This way we visit N/2 nodes, and thus the time complexity is
          O(N).
        - Space complexity: is the same as time complexity here O(N).
        """
        if not root:
            return 0

        # Initialize BFS queue
        bfsQueue: deque[Tuple[TreeNode, int]] = deque([])
        # Add root node to bfsQueue (null check was done above)
        bfsQueue.append((root, 1))

        while len(bfsQueue) != 0:
            # Get node in front of the queue (Use popleft to get first inserted element)
            (node, depth) = bfsQueue.popleft()
            # Case 1: If node is a leaf node, you found the minimum depth, early terminate and return
            if not node.left and not node.right:
                return depth
            # Case 2: If left subtree exists (and right subtree doesn't, checked above)
            if node.left:
                bfsQueue.append((node.left, depth+1))
            # Case 3: If right subtree exists (and left subtree doesn't, checked above)
            if node.right:
                bfsQueue.append((node.right, depth+1))


if __name__ == "__main__":
    # Object
    minimumDepthOfBinaryTree: MinimumDepthOfBinaryTree = MinimumDepthOfBinaryTree()

    """
    Example 1:
    Input: root = [3,9,20,null,null,15,7]
    Output: 2
    
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
    print(f"Minimum depth of tree rooted at Node {node3.val} using recursive solution: "
          f"{minimumDepthOfBinaryTree.minDepthRecursive(node3)}")
    print(f"Minimum depth of tree rooted at Node {node3.val} using iterative BFS solution: "
          f"{minimumDepthOfBinaryTree.minDepthBfsIterative(node3)}")
    """
    Minimum depth of tree rooted at Node 3 using recursive solution: 2
    Minimum depth of tree rooted at Node 3 using iterative BFS solution: 2
    """

    """
    Example 2:
    Input: root = [10,null,11,null,12,null,13,null,14]
    Output: 5
     => Careful: minDepth is not 1. There is 1 leaf node, 14, and its depth is 5.
    
         10
          \
           11
            \
             12
              \
               13
                \
                 14

    """
    # Nodes
    node10: TreeNode = TreeNode(10)
    node11: TreeNode = TreeNode(11)
    node12: TreeNode = TreeNode(12)
    node13: TreeNode = TreeNode(13)
    node14: TreeNode = TreeNode(14)
    # Edges
    node10.right = node11
    node11.right = node12
    node12.right = node13
    node13.right = node14
    # Test
    print(f"Minimum depth of tree rooted at Node {node10.val} using recursive solution: "
          f"{minimumDepthOfBinaryTree.minDepthRecursive(node10)}")
    print(f"Minimum depth of tree rooted at Node {node10.val} using iterative BFS solution: "
          f"{minimumDepthOfBinaryTree.minDepthBfsIterative(node10)}")
    """
    Minimum depth of tree rooted at Node 10 using recursive solution: 5
    Minimum depth of tree rooted at Node 10 using iterative BFS solution: 5
    """