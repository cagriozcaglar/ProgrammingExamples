"""
Lowest Common Ancestor of a Binary Tree III:
LCA of a BT, when parent pointers are available
"""
from typing import List

# Definition for a Node.
class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

class Solution:
    # LCA with parent pointer
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        # 1. Generate parent list for both p and q
        pList = self.generateParentList(p)
        qList = self.generateParentList(q)
        # self.printNodeValuesInList(pList)
        # self.printNodeValuesInList(qList)

        # 2. Depth difference: truncate longer ancestor list to length of shorter ancestor list
        # Denote p as longer path
        if len(qList) > len(pList):
            pList, qList = pList, qList
        # Trim longer list p
        pList = pList[len(pList)-len(qList):]

        # 3. Compare node values(!) one-by-one from lowest level to root. When matched, LCA
        for index in range(len(pList)):
            if pList[index].val == qList[index].val:
                return pList[index]

    # Given a node, generate its ancestor list by traversing its ancestors to root
    def generateParentList(self, node: 'Node') -> List['Node']:
        ancestorList = []
        while node:
            ancestorList.append(node)
            node = node.parent
        return ancestorList

    # Node list printer method
    def printNodeValuesInList(self, pList: List['Node']):
        values = [node.val for node in pList]
        print(f"values: {values}")