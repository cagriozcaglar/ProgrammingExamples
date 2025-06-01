'''
Leetcode 994: Rotting Oranges

You are given an m x n grid where each cell can have one of three values:
# 0 representing an empty cell,
# 1 representing a fresh orange, or
# 2 representing a rotten orange.

If a fresh orange is adjacent (4-directionally) to a rotten orange, it becomes rotten as well.

You need to return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.
'''
from collections import deque
from enum import Enum
from typing import List

class Cell(Enum):
    EMPTY = 0
    FRESH = 1
    ROTTEN = 2

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # Get the number of rows and columns of the grid.
        rows, cols = len(grid), len(grid[0])
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        def is_within_bounds(row: int, col: int) -> bool:
            return 0 <= row < rows and 0 <= col < cols

        # Initialize a queue for BFS and a counter for fresh oranges.
        queue = deque()
        fresh_count = 0

        # Go through each cell in the grid.
        for i in range(rows):
            for j in range(cols):
                # If we find a rotten orange, add its position to the queue.
                if grid[i][j] == Cell.ROTTEN.value:
                    queue.append((i, j))
                # If it's a fresh orange, increment the fresh_count.
                elif grid[i][j] == Cell.FRESH.value:
                    fresh_count += 1

        # Track the number of minutes passed.
        minutes_passed = 0

        # Perform BFS until the queue is empty or there are no fresh oranges left.
        while queue and fresh_count > 0:
            # Increment minutes each time we start a new round of BFS.
            minutes_passed += 1

            # Loop over all the rotten oranges at the current minute.
            for _ in range(len(queue)):
                row, col = queue.popleft()

                # Check the adjacent cells in all four directions.
                for d_r, d_c in directions:
                    new_row, new_col = row + d_r, col + d_c

                    # If the adjacent cell has a fresh orange, rot it.
                    if is_within_bounds(new_row, new_col) and \
                       grid[new_row][new_col] == Cell.FRESH.value:
                        fresh_count -= 1
                        grid[new_row][new_col] = 2
                        queue.append((new_row, new_col))

        # If there are no fresh oranges left, return the minutes passed.
        # Otherwise, return -1 because some oranges will never rot.
        return minutes_passed if fresh_count == 0 else -1