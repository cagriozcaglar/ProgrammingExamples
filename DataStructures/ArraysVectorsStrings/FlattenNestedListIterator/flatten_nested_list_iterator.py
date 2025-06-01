'''
Leetcode 341: Flatten Nested List Iterator

Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Input: [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,1,2,1,1].

Example 2:
Input: [1,[4,[6]]]
Output: [1,4,6]
Explanation: By calling next repeatedly until hasNext returns false, 
             the order of elements returned by next should be: [1,4,6].

'''
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
from typing import List

class NestedInteger:
   def isInteger(self) -> bool:
       """
       @return True if this NestedInteger holds a single integer, rather than a nested list.
       """

   def getInteger(self) -> int:
       """
       @return the single integer that this NestedInteger holds, if it holds a single integer
       Return None if this NestedInteger holds a nested list
       """

   def getList(self) -> List['NestedInteger']:
       """
       @return the nested list that this NestedInteger holds, if it holds a nested list
       Return None if this NestedInteger holds a single integer
       """


# Solution 5: Using Generators
# Constructor: O(1)
# next(): O(L/N) or O(1)
# hasNext(): O(L/N) or O(1)
# Space: O(D), due to recursive call to int_generator in constructor
class NestedIterator:
    def __init__(self, nestedList: List[NestedInteger]):
        # Generator object from the generator function, passing in nestedList
        self.generator = self.int_generator(nestedList)
        # All values are placed here before returned
        self.peeked = None

    def int_generator(self, nested_list: List[NestedInteger]):
        for nested in nested_list:
            if nested.isInteger():
                yield nested.getInteger()
            else:
                yield from self.int_generator(nested.getList())

    def next(self) -> int:
        if not self.hasNext():
            return None
        # Return the value of self.peeked, also clear it
        next_integer, self.peeked = self.peeked, None
        return next_integer

    def hasNext(self) -> bool:
        if self.peeked is not None:
            return True
        try:  # Get new integer out of generator
            self.peeked = next(self.generator)
            return True
        except:  # Generator is finished so raised StopIteration
            return False


# Solution 1: Make a Flat List with Recursion
# Constructor: O(N+L), N: num integers, L: number of lists
# next(): O(1)
# hasNext(): O(1)
# Space: O(N+D), (D, because of recursive call stack for flatten_list method)
class NestedIterator1:
    def __init__(self, nestedList: List[NestedInteger]):
        def flatten_list(nestedList: List[NestedInteger]) -> None:
            for value in nestedList:
                if value.isInteger():
                    self.values.append(value.getInteger())
                else:
                    flatten_list(value.getList())

        self.values = []
        self.position = -1  # Pointer to previous returned
        flatten_list(nestedList)

    def next(self) -> int:
        self.position += 1
        return self.values[self.position]

    def hasNext(self) -> bool:
        return self.position + 1 < len(self.values)


# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())