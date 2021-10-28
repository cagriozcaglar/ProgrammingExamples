"""

Note: This is Leetcode question 429: https://leetcode.com/problems/n-ary-tree-level-order-traversal/

Given an n-ary tree, return the level order traversal of its nodes' values.
Nary-Tree input serialization is represented in their level order traversal, each group of children is separated by the
null value (See examples).

Example 1:
Input: root = [1,null,3,2,4,null,5,6]
Output: [[1],[3,2,4],[5,6]]

Example 2:
Input: root = [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]
Output: [[1],[2,3,4,5],[6,7,8,9,10],[11,12,13],[14]]

Constraints:
 - The height of the n-ary tree is less than or equal to 1000
 - The total number of nodes is between [0, 104]
"""
from typing import List
from collections import deque


class Node:
    """
    Definition for a Node.
    """
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


class NaryTreeLevelOrderTraversal:
    @staticmethod
    def levelOrderWithLists(root: 'Node') -> List[List[int]]:
        """
        This is simplified BFS, with lists to keep nodes at each level, no queue used.
        We make a new list on each iteration instead of using a single queue.

        - Time complexity : O(n), where nn is the number of nodes.
        - Space complexity : O(n). Same as above, we always have lists containing levels of nodes.
        :param root:
        :return:
        """
        if root is None:
            return []

        # Initialize list of lists to return
        nodeListByLevel: List[List[int]] = []

        currentLevelList: List[Node] = []
        currentLevelList.append(root)
        while currentLevelList:
            # Add the list of node values from previous level
            nodeListByLevel.append([node.val for node in currentLevelList])
            # Start current level
            parents: List[Node] = currentLevelList
            # Initialize current level list of nodes
            currentLevelList = []
            for parent in parents:
                if parent.children:  # NOTE: Do not forget to check if parent has children, before iterating on it next
                    for child in parent.children:
                        currentLevelList.append(child)
        return nodeListByLevel

    @staticmethod
    def levelOrderwithBfsQueue(root: 'Node') -> List[List[int]]:
        """
        BFS, keep track of nodes at each level

        - Time complexity: O(n), where n is the number of nodes. Each node is getting added to the queue, removed from
          the queue, and added to the result exactly once.
        - Space complexity: O(n). We are using a queue to keep track of nodes we still need to visit the children of.
          At most, the queue will have 2 layers of the tree on it at any given time. In the worst case, this is all of
          the nodes. In the best case, it is just 1 node (if we have a tree that is equivalent to a linked list). The
          average case is difficult to calculate without knowing something of the trees we can expect to see, but in
          balanced trees, half or more of the nodes are often in the lowest 2 layers. So we should go with the worst
          case of O(n), and know that the average case is probably similar.
        :param root:
        :return:
        """
        if root is None:
            return []

        # Initialize list of lists to return
        nodeListByLevel: List[List[int]] = []

        # Create BFS queue, push root to it
        bfsQueue: deque[Node] = deque([root])

        while bfsQueue:
            currentLevelList = []
            # Get current size of bfsQueue, pop all of these nodes, put node values of them to currentLevelList, and
            # extend the nodes in bfsQueue with nodes children.
            bfsQueueCurrentSize = len(bfsQueue)
            for i in range(bfsQueueCurrentSize):
                node: Node = bfsQueue.popleft()
                currentLevelList.append(node.val)
                # NOTE: Before extending bfsQueue with node children list next, check if node.children is not None
                # Because if it is None, then the next statement "bfsQueue.extend(node.children)" returns error saying
                # "TypeError: 'NoneType' object is not iterable"
                if node.children is not None:
                    # NOTE: When adding element to a list, use append() method (see above). When extending a list with
                    # another list, use extend() method (see below)
                    bfsQueue.extend(node.children)
            nodeListByLevel.append(currentLevelList)
        return nodeListByLevel


if __name__ == "__main__":
    """
    Example 1:
    Input: root = [1,null,3,2,4,null,5,6]
    Output: [[1],[3,2,4],[5,6]]
    
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
    print(f"Level-order traversal of n-ary tree rooted at {node1.val} using lists: "
          f"{NaryTreeLevelOrderTraversal.levelOrderWithLists(node1)}")
    print(f"Level-order traversal of n-ary tree rooted at {node1.val} using BFS queue: "
          f"{NaryTreeLevelOrderTraversal.levelOrderwithBfsQueue(node1)}")

    """
    Level-order traversal of n-ary tree rooted at 1 using lists: [[1], [3, 2, 4], [5, 6]]
    Level-order traversal of n-ary tree rooted at 1 using BFS queue: [[1], [3, 2, 4], [5, 6]]
    """