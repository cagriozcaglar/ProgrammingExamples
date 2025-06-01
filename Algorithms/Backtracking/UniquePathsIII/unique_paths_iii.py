'''
Leetcode 980: Unique Paths III

You are given an m x n integer array grid where grid[i][j] could be:

1 representing the starting square. There is exactly one starting square.
2 representing the ending square. There is exactly one ending square.
0 representing empty squares we can walk over.
-1 representing obstacles that we cannot walk over.
Return the number of 4-directional walks from the starting square to the ending square, that walk over every non-obstacle square exactly once.
'''
from typing import List

class Solution:
    def uniquePathsIII(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        directions = [ [0, 1], [0, -1], [1, 0], [-1, 0] ]

        def is_within_bounds(row: int, col: int) -> bool:
            return 0 <= row < rows and 0 <= col < cols

        # 1. Find start cell, and all non_obstacle cell count
        non_obstacles = 0
        start_row, start_col = 0, 0
        for row in range(0, rows):
            for col in range(0, cols):
                cell = grid[row][col]
                if cell >= 0:  # 0 (empty), 1 (start), or 2 (end)
                    non_obstacles += 1
                if cell == 1:
                    start_row, start_col = row, col

        # Backtracking method
        def backtrack(row: int, col: int, remain: int) -> None:
            # # We need to modify external variable path_count
            nonlocal path_count

            # Base case: Termination of backtracking
            # a) Cell is end cell, b) remaining cell is 1
            if grid[row][col] == 2 and remain == 1:
                path_count += 1
                return

            # Make move: Mark the square as visited. Case: 0, 1, 2
            temp = grid[row][col]
            grid[row][col] = -4  # visited
            remain -= 1

            # Backtrack: Explore 4 directions around
            for d_row, d_col in directions:
                next_row, next_col = row + d_row, col + d_col

                # Invalid coordinate, or "obstacle or visited square"
                # Continue to next for loop iteration
                if not is_within_bounds(next_row, next_col) or grid[next_row][next_col] < 0:
                    continue

                backtrack(next_row, next_col, remain)

            # Unmake move
            grid[row][col] = temp

        # Count of paths as the final result
        path_count = 0
        # Start backtracking here
        backtrack(start_row, start_col, non_obstacles)

        return path_count