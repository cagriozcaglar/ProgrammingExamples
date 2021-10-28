"""
116. Populating Next Right Pointers in Each Node
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The
binary tree has the following definition:
    struct Node {
      int val;
      Node *left;
      Node *right;
      Node *next;
    }
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be
set to NULL.
Initially, all next pointers are set to NULL.
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return root

        # BFS queue
        bfsQueue = deque([root])

        while bfsQueue:
            # Record the number of nodes at this level
            size = len(bfsQueue)
            # Iterate over all the nodes on the current level
            for i in range(size):
                # Pop from front of the queue
                node = bfsQueue.popleft()
                # Check this, because the bfsQueue will contain nodes from
                # 2 levels at most at any point in time
                if i < size-1:
                    # Point the node.next pointer next element in bfsQueue
                    node.next = bfsQueue[0]

                # Add children to back of the queue
                if node.left:
                    bfsQueue.append(node.left)
                if node.right:
                    bfsQueue.append(node.right)
        # Tree has been modified, return the root
        return root