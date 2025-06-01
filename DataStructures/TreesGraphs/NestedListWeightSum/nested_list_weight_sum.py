'''
Leetcode 339: Nested List Weight Sum

You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists.

The depth of an integer is the number of lists that it is inside of. For example, the nested list [1,[2,2],[[3],2],1] has each integer's value set to its depth.

Return the sum of each integer in nestedList multiplied by its depth.

Example 1:
Input: nestedList = [[1,1],2,[1,1]]
Output: 8
Explanation: Four 1's with a weight of 1, one 2 with a weight of 2.
1*1 + 1*1 + 2*2 + 1*1 + 1*1 = 8

Example 2:
Input: nestedList = [1,[4,[6]]]
Output: 17
Explanation: One 1 at depth 3, one 4 at depth 2, and one 6 at depth 1.
1*3 + 4*2 + 6*1 = 17
'''

"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation
"""
class NestedInteger:
   def __init__(self, value=None):
       """
       If value is not specified, initializes an empty list.
       Otherwise initializes a single integer equal to value.
       """

   def isInteger(self):
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       :rtype bool
       """

   def add(self, elem):
       """
       Set this NestedInteger to hold a nested list and adds a nested integer elem to it.
       :rtype void
       """

   def setInteger(self, value):
       """
       Set this NestedInteger to hold a single integer equal to value.
       :rtype void
       """

   def getInteger(self):
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       The result is undefined if this NestedInteger holds a nested list
       :rtype int
       """

   def getList(self):
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       The result is undefined if this NestedInteger holds a single integer
       :rtype List[NestedInteger]
       """

from typing import List
from collections import deque

class Solution:
    # Solution 1: BFS
    # Time complexity : O(N).
    # Space complexity : O(N).
    # The worst-case for space complexity in BFS occurs where most of the elements are in a single layer, for example, a flat list such as [1, 2, 3, 4, 5] as all of the elements must be put on the queue at the same time. Therefore, this approach also has a worst-case space complexity of O(N).
    def depthSumBfs(self, nestedList: List[NestedInteger]) -> int:
        depth = 1
        total = 0
        bfs_queue = deque(nestedList)

        while bfs_queue:
            level_size = len(bfs_queue)
            for i in range(level_size):
                value = bfs_queue.popleft()
                if value.isInteger():
                    total += value.getInteger() * depth
                else:
                    for element in value.getList():
                        bfs_queue.append(element)
            depth += 1

        return total

    # Solution 1: DFS
    # Time complexity : O(N).
    # Space complexity : O(N).
    # In terms of space, at most O(D) recursive calls are placed on the stack, where D is the maximum level of nesting in the input. For example, D=2 for the input [[1,1],2,[1,1]], and D=3 for the input [1,[4,[6]]]. In the worst case, D=N, (e.g. the list [[[[[[]]]]]]) so the worst-case space complexity is O(N).
    def depthSumDfs(self, nestedList: List[NestedInteger]) -> int:
        def dfs(nested_list: List[NestedInteger], depth: int):
            total = 0
            for value in nested_list:
                if value.isInteger():
                    total += value.getInteger() * depth
                else:
                    total += dfs(value.getList(), depth + 1)
            return total

        return dfs(nestedList, 1)