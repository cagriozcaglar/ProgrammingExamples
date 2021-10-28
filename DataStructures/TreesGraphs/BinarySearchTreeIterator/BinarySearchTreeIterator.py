"""
173. Binary Search Tree Iterator
Implement the BSTIterator class that represents an iterator over the in-order traversal of a binary search tree (BST):
1) BSTIterator(TreeNode root) Initializes an object of the BSTIterator class. The root of the BST is given as part of
   the constructor. The pointer should be initialized to a non-existent number smaller than any element in the BST.
2) boolean hasNext() Returns true if there exists a number in the traversal to the right of the pointer, otherwise
   returns false.
3) int next() Moves the pointer to the right, then returns the number at the pointer.
"""
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.stack = []
        self._leftmostInorder(root)

    def _leftmostInorder(self, root):
        # For a given node, add all elements in the leftmost branch of the tree under it to the stack
        while root:
            self.stack.append(root)
            root = root.left

    def next(self) -> int:
        topNode = self.stack.pop()
        # Maintain invariant. If node has right child, call the helper function for the right child
        if topNode.right:
            self._leftmostInorder(topNode.right)
        return topNode.val

    def hasNext(self) -> bool:
        return len(self.stack) > 0

# Your BSTIterator object will be instantiated and called as such:
# obj = BSTIterator(root)
# param_1 = obj.next()
# param_2 = obj.hasNext()