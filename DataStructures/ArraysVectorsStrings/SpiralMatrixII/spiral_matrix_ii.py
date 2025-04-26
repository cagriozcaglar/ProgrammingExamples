'''
Leetcode 59: Spiral Matrix II

Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

Example 1:
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]

Example 2:
Input: n = 1
Output: [[1]]
'''
from typing import List

# Time: O(n^2)
# Space: O(1)
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        # Initialize matrix
        NOT_VISITED = -1
        matrix = [[NOT_VISITED] * n for _ in range(n)]

        # Directions
        directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        # Initialize row, col, num, dir_index
        row, col = 0, 0
        num = 1
        dir_index = 0

        def within_bounds(row: int, col: int) -> bool:
            return 0 <= row < n and 0 <= col < n

        while num <= n ** 2:
            matrix[row][col] = num

            # Next direction
            d_row, d_col = directions[dir_index]

            if not (within_bounds(row + d_row, col + d_col)) or \
                matrix[row + d_row][col + d_col] == NOT_VISITED:
                # Change direction
                dir_index = (dir_index + 1) % 4

            d_row, d_col = directions[dir_index]
            row, col = row + d_row, col + d_col
            num += 1

        return matrix