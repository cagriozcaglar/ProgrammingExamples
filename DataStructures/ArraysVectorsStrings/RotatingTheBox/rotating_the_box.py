'''
Leetcode 1861: Rotating the Box

You are given an m x n matrix of characters box representing a side-view of a box. Each cell of the box is one of the following:

A stone '#'
A stationary obstacle '*'
Empty '.'

The box is rotated 90 degrees clockwise, causing some of the stones to fall due to gravity. Each stone falls down until it lands on an obstacle, another stone, or the bottom of the box. Gravity does not affect the obstacles' positions, and the inertia from the box's rotation does not affect the stones' horizontal positions.

It is guaranteed that each stone in box rests on an obstacle, another stone, or the ground.

Return an n x m matrix representing the box after the rotation described above.
'''
from typing import List

class Solution:
    '''
    Time complexity: O(m×n)

    Similar to the first approach, the rotation operation takes O(m×n) time. The gravity effect is now implemented using two nested loops instead of three. The outer loop iterates over the m columns, and for each column, the inner loop processes all n elements. As a result, the total time complexity of the algorithm remains O(m×n).

    Space complexity: O(m×n)

    Once again, we avoid modifying the input directly by creating a second grid, result, of size n×m. However, if we were allowed to modify the input in place, the space complexity could be reduced to O(1).    
    '''

    def rotateTheBox(self, boxGrid: List[List[str]]) -> List[List[str]]:
        m, n = len(boxGrid), len(boxGrid[0])
        rotatedBox = [[''] * m for _ in range(n)]

        # Rotate the box: rotatedBox[i][j] = box[m-1-j][i]
        for row in range(n):
            for col in range(m):
                rotatedBox[row][col] = boxGrid[m-1-col][row]

        # Apply gravity
        for j in range(m):  # columns
            lowest_row_with_empty_cell = n - 1  # bottom row
            # Process each cell in column j, bottom up
            for i in range(n-1, -1, -1):
                # Found a stone - let it fall to the lowest empty cell
                if rotatedBox[i][j] == "#":
                    rotatedBox[i][j] = "."
                    rotatedBox[lowest_row_with_empty_cell][j] = "#"
                    lowest_row_with_empty_cell -= 1
                # Found obstacle - reset lowest_row_with_empty_cell to the row above it
                if rotatedBox[i][j] == "*":
                    lowest_row_with_empty_cell = i - 1

        return rotatedBox