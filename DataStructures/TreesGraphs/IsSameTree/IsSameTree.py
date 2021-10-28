"""
Note: This is Leetcode question 100: https://leetcode.com/problems/same-tree/

Given two binary trees, write a function to check if they are the same or not.
Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

Example 1:
Input:     1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]
Output: true

Example 2:
Input:     1         1
          /           \
         2             2

        [1,2],     [1,null,2]
Output: false

Example 3:
Input:     1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]
Output: false
"""


class TreeNode:
    """
    Definition for a binary tree node.
    """
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class IsSameTree:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        """
        Recursive solution.
        Three cases:
        1) p and q are not Null: Then, check node value equality (p.val == q.val), and recursively check left tree
           equality (self.isSameTree(p.left, q.left)) and right tree equality (self.isSameTree(p.right, q.right)).
        2) p and q are both Null: Nodes are equal, return True.
        3) Else (one of p and q is Null, the other one is non-null): Nodes are not equal, return False.

        - Time complexity: O(N), where N is a number of nodes in the tree, since one visits each node exactly once.
        - Space complexity: O(log(N)) in the best case of completely balanced tree and O(N) in the worst case of
          completely unbalanced tree, to keep a recursion stack.
        """
        if p is not None and q is not None:
            return (p.val == q.val) and \
                   self.isSameTree(p.left, q.left) and \
                   self.isSameTree(p.right, q.right)
        elif p is None and q is None:
            return True
        else:
            return False


if __name__ == "__main__":
    # IsSameTree().isSameTree()
    """
    Example 1:
    Input:     1         1
              / \       / \
             2   3     2   3
    
            [1,2,3],   [1,2,3]
    Output: true
    """
    # Tree 1
    # Nodes
    node1: TreeNode = TreeNode(1)
    node2: TreeNode = TreeNode(2)
    node3: TreeNode = TreeNode(3)
    # Edges
    node1.left = node2
    node1.right = node3

    # Tree 2
    node1Again = node1

    # Test
    print(f"Are the trees rooted at Node {node1.val} and Node {node1Again.val} the same?: "
          f"{IsSameTree().isSameTree(node1, node1Again)}")
    """
    Are the trees rooted at Node 1 and Node 1 the same?: True
    """

    """
    Example 2:
    Input:     1         1
              /           \
             2             2
    
            [1,2],     [1,null,2]
    Output: false
    """
    # Tree 1:
    node1Example2Tree1 = node1
    node2Example2Tree1 = node2
    node1Example2Tree1.left = node2Example2Tree1
    # Tree 2:
    node1Example2Tree2 = TreeNode(1)
    node2Example2Tree2 = TreeNode(2)
    node1Example2Tree2.right = node2Example2Tree2

    # Test
    print(f"Are the trees rooted at Node {node1Example2Tree1.val} and Node {node1Example2Tree2.val} the same?: "
          f"{IsSameTree().isSameTree(node1Example2Tree1, node1Example2Tree2)}")
    """
    Are the trees rooted at Node 1 and Node 1 the same?: False
    """