"""
Number Of Islands.
Solution with DFS.
"""
from typing import List
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        # Initialization
        nrows, ncols = len(grid), len(grid[0])
        islandCount = 0
        directions = [ [0, 1], [0, -1], [1, 0], [-1, 0] ]
        # No need for visited map or adjacency list

        def isWithinBounds(row, col) -> bool:
            return 0 <= row < nrows and 0 <= col < ncols

        def dfs(row, col) -> None:
            # Out of bounds, or not a land
            if not isWithinBounds(row, col) or grid[row][col] == "0":
                return
            # Mark as not a land (indicator of having visited the land)
            grid[row][col] = "0"
            # Check neighbours
            for drow, dcol in directions:
                newRow, newCol = row + drow, col + dcol
                dfs(newRow, newCol)

        for row in range(nrows):
            for col in range(ncols):
                if grid[row][col] == "1":  # Land
                    islandCount += 1
                    dfs(row, col)

        return islandCount