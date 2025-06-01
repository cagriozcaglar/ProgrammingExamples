'''
Leetcode 3071: Minimum Operations to write letter y on grid

You are given a 0-indexed n x n grid where n is odd, and grid[r][c] is 0, 1, or 2.

We say that a cell belongs to the Letter Y if it belongs to one of the following:

The diagonal starting at the top-left cell and ending at the center cell of the grid.
The diagonal starting at the top-right cell and ending at the center cell of the grid.
The vertical line starting at the center cell and ending at the bottom border of the grid.
The Letter Y is written on the grid if and only if:

All values at cells belonging to the Y are equal.
All values at cells not belonging to the Y are equal.
The values at cells belonging to the Y are different from the values at cells not belonging to the Y.
Return the minimum number of operations needed to write the letter Y on the grid given that in one operation you can change the value at any cell to 0, 1, or 2.
'''

'''
We use two arrays of length 3, cnt1 and cnt2, to record the counts of cell values that belong to Y and do not belong to Y, respectively. Then we enumerate i and j, which represent the values of cells that belong to Y and do not belong to Y, respectively, to calculate the minimum number of operations.

The time complexity is O(n^2), where n is the size of the matrix. The space complexity is O(1).
'''
from collections import defaultdict
from typing import List

class Solution:
    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])

        # cnt1 and cnt2, to record the counts of cell values that belong to Y and do not belong to Y
        count_in_y, count2_out_y = defaultdict(int), defaultdict(int)

        for row_index, row in enumerate(grid):
            for col_index, cell_value in enumerate(row):
                in_top_left_diagonal = (row_index == col_index) and row_index <= n // 2
                in_top_right_diagonal = (row_index + col_index == n-1) and row_index <= n // 2
                in_bottom_middle_line = (col_index == n // 2) and row_index >= n // 2
                if in_top_left_diagonal or in_top_right_diagonal or in_bottom_middle_line:
                    count_in_y[cell_value] += 1
                else:
                    count2_out_y[cell_value] += 1

        return min (n**2 - count_in_y[i] - count2_out_y[j] for i in range(3) for j in range(3) if i !=j)
