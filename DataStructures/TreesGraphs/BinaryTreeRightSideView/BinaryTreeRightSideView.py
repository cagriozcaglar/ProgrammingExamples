"""
Binary Tree Right Side View
BFS on binary tree, very minor modification
"""
from typing import Optional, List
from collections import deque
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        bfsQueue = deque([root])
        rightSide = []

        while bfsQueue:
            # Get length at entrance to every level
            levelLength = len(bfsQueue)
            # Iterate over all nodes
            for i in range(levelLength):
                node = bfsQueue.popleft()
                # If rightmost element, add to result
                if i == levelLength-1:
                    rightSide.append(node.val)
                # Add child nodes to BFS queue
                if node.left:
                    bfsQueue.append(node.left)
                if node.right:
                    bfsQueue.append(node.right)
        return rightSide