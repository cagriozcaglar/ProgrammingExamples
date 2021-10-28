"""
536. Construct Binary Tree from String
You need to construct a binary tree from a string consisting of parenthesis and integers.

The whole input represents a binary tree. It contains an integer followed by zero, one or two pairs of parenthesis. The
integer represents the root's value and a pair of parenthesis contains a child binary tree with the same structure.

You always start to construct the left child node of the parent first if it exists.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def str2tree(self, s: str) -> Optional[TreeNode]:
        return self._str2treeInternal(s,0)[0]

    def _getNumber(self, s: str, index: int) -> (int, int):
        isNegative = False

        # A negative number
        if s[index] == "-":
            isNegative = True
            index += 1

        number = 0
        while index < len(s) and s[index].isdigit():
            number = number*10 + int(s[index])
            index += 1

        return number if not isNegative else -number, index

    def _str2treeInternal(self, s: str, index: int) -> (TreeNode, int):
        if index == len(s):
            return None, index

        # Start of tree will contain number representing root
        value, index = self._getNumber(s, index)
        node = TreeNode(value)

        # If there is any data left, check for first subtree,
        # which will always be the left child
        if index < len(s) and s[index] == "(":
            node.left, index = self._str2treeInternal(s. index+1)
        # Indicates a right child
        if node.left and index < len(s) and s[index] == "(":
            node.right, index = self._str2treeInternal(s, index+1)

        return node, index+1 if index < len(s) and s[index] == ")" else index