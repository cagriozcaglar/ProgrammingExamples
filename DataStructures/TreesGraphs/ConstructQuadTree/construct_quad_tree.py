'''
Leetcode 427: Construct Quad Tree

Given a n * n matrix grid of 0's and 1's only. We want to represent the grid with a Quad-Tree.

Return the root of the Quad-Tree representing the grid.

A Quad-Tree is a tree data structure in which each internal node has exactly four children. Besides, each node has two attributes:

val: True if the node represents a grid of 1's or False if the node represents a grid of 0's. Notice that you can assign the val to True or False when isLeaf is False, and both are accepted in the answer.
isLeaf: True if the node is a leaf node on the tree or False if the node has four children.
class Node {
    public boolean val;
    public boolean isLeaf;
    public Node topLeft;
    public Node topRight;
    public Node bottomLeft;
    public Node bottomRight;
}
We can construct a Quad-Tree from a two-dimensional area using the following steps:

If the current grid has the same value (i.e all 1's or all 0's) set isLeaf True and set val to the value of the grid and set the four children to Null and stop.
If the current grid has different values, set isLeaf to False and set val to any value and divide the current grid into four sub-grids as shown in the photo.
Recurse for each of the children with the proper sub-grid.
'''

# From multiple sources
# 1. Leetcode Editorial
# 2. https://leetcode.ca/2017-01-30-427-Construct-Quad-Tree/
# 3. https://walkccc.me/LeetCode/problems/427/#__tabbed_1_3

from itertools import product
from typing import List

# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

class Solution:
    # Solution 1: Recursion
    # Time complexity: O(N^2 * log(N))
    # Space complexity: O(logN) (Because of recursive call stack)
    def construct(self, grid: List[List[int]]) -> 'Node':
        # Check if grid is homogeneous (all 0s or 1s)
        def same_value_grid(x1: int, y1: int, length: int) -> bool:
            # Some from: https://walkccc.me/LeetCode/problems/427/#__tabbed_1_3
            # IMPORTANT: do not ise grid[x1: x2][y1: y2], its syntax is not correct.
            return all(grid[x1][y1] == grid[x][y] for x, y in product(range(x1, x1+length), range(y1, y1+length)))

        def construct_quad_tree(x1: int, y1: int, length: int) -> 'Node':
            # Node is homogeneous, create a leaf node
            if( same_value_grid(x1, y1, length) ):
                return Node(
                    val=(grid[x1][y1] == 1),
                    isLeaf=True,
                )
            # Node is not homogeneous, create four child nodes
            else:
                root: Node = Node(False, False)
                # IMPORTANT: Divide length using //, otherwise it will return float, 
                # and this will cause error, because range() only accepts an integer, not float
                half_length = length // 2 
                # Recursive calls
                root.topLeft = construct_quad_tree(x1, y1, half_length)
                root.topRight = construct_quad_tree(x1, y1+half_length, half_length)
                root.bottomLeft = construct_quad_tree(x1+half_length, y1, half_length)
                root.bottomRight = construct_quad_tree(x1+half_length, y1+half_length, half_length)
                return root

        return construct_quad_tree(0, 0, len(grid))
