'''
Leetcode 2768: Number of Black Blocks

You are given two integers m and n representing the dimensions of a 0-indexed m x n grid.

You are also given a 0-indexed 2D integer matrix coordinates, where coordinates[i] = [x, y] indicates that the cell with coordinates [x, y] is colored black. All cells in the grid that do not appear in coordinates are white.

A block is defined as a 2 x 2 submatrix of the grid. More formally, a block with cell [x, y] as its top-left corner where 0 <= x < m - 1 and 0 <= y < n - 1 contains the coordinates [x, y], [x + 1, y], [x, y + 1], and [x + 1, y + 1].

Return a 0-indexed integer array arr of size 5 such that arr[i] is the number of blocks that contains exactly i black cells.

Example 1:
Input: m = 3, n = 3, coordinates = [[0,0]]
Output: [3,1,0,0,0]
Explanation: The grid looks like this:
There is only 1 block with one black cell, and it is the block starting with cell [0,0].
The other 3 blocks start with cells [0,1], [1,0] and [1,1]. They all have zero black cells. 
Thus, we return [3,1,0,0,0]. 
'''
from typing import List, Dict, Tuple
from collections import defaultdict

class Solution:
    def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
        # Block is uniquely identified by (x,y) coord of top-left corner
        block_black_counts: Dict[Tuple[int, int], int] = defaultdict(int)

        # For a given black cell, it can belong to 4 blocks defined by top-left
        directions = [ [0, 0], [-1, 0], [0, -1], [-1, -1] ]

        def block_top_left_within_bounds(i: int, j: int):
            return 0 <= i < m - 1 and 0 <= j < n - 1

        for coord in coordinates:
            x, y = coord[0], coord[1]
            for dir_pair in directions:
                # top-left corner coordinate, uniquely identifying a block
                i, j = x + dir_pair[0], y + dir_pair[1]
                if block_top_left_within_bounds(i, j):
                    block_black_counts[(i,j)] += 1

        # Iterate over block black counts and update arr
        arr = [0] * 5
        for block_top_left, count in block_black_counts.items():
            arr[count] += 1

        # block_black_counts doesn't list blocks with 0 black cells
        # We derive it as follows: There are (m-1)*(n-1) blocks.
        # len(block_black_counts) returns number of blocks with at least one black cell.
        # Number of blocks with 0 black cell is: (m-1)*(n-1) - len(block_black_counts)
        # arr[0] = (m-1)*(n-1) - len(block_black_counts)
        arr[0] = (m-1)*(n-1) - sum(arr[i] for i in range(1,5))

        return arr