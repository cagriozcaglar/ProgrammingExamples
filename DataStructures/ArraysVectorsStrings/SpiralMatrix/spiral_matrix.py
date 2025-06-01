'''
Leetcode 54: Spiral Matrix
Given an m x n matrix, return all elements of the matrix in spiral order.
'''
from typing import List

class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        rows, cols = len(matrix), len(matrix[0])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        visited = [[False] * cols for _ in range(rows)]

        def is_within_bounds(row: int, col: int) -> bool:
            return 0 <= row < rows and 0 <= col < cols

        result = []

        dir_index = 0
        cell = (0, 0)

        while len(result) < rows * cols:
            (row, col) = cell
            result.append(matrix[row][col])
            visited[row][col] = True

            new_row, new_col = row + directions[dir_index][0], col + directions[dir_index][1]

            if not is_within_bounds(new_row, new_col) or \
               visited[new_row][new_col]:
                dir_index = (dir_index + 1) % 4
                new_row, new_col = row + directions[dir_index][0], col + directions[dir_index][1]

            cell = (new_row, new_col)

        return result