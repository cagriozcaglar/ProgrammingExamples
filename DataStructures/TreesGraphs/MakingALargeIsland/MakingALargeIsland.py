"""
827. Making A Large Island
You are given an n x n binary matrix grid. You are allowed to change at most one 0 to be 1.
Return the size of the largest island in grid after applying this operation.
An island is a 4-directionally connected group of 1s.
"""
from typing import List

class Solution:
    def largestIsland(self, grid: List[List[int]]) -> int:
        N = len(grid)
        directions = [ [0,1], [0, -1], [1, 0], [-1, 0] ]

        def isWithinBounds(row: int, col: int) -> bool:
            return 0 <= row < N and 0 <= col < N

        # dfs later
        def dfs(row, col, index):
            ans = 1
            grid[row][col] = index
            for dRow, dCol in directions:
                newRow, newCol = row+dRow, col+dCol
                if isWithinBounds(newRow, newCol) and grid[newRow][newCol] == 1:
                    ans += dfs(newRow, newCol, index)
            return ans

        area = {}
        index = 2
        for row in range(N):
            for col in range(N):
                if grid[row][col] == 1:
                    area[index] = dfs(row, col, index)
                    index += 1

        # Before we change 0's to 1's,
        # The last [0] is for the case where all values are 0s in grid, then default to [0]
        ans = max(area.values() or [0])

        # Replace 0's with 1's calculate combined area
        for row in range(N):
            for col in range(N):
                if grid[row][col] == 0:
                    seen = set([])
                    for dRow, dCol in directions:
                        newRow, newCol = row+dRow, col+dCol
                        if isWithinBounds(newRow, newCol) and grid[newRow][newCol] > 1:
                            seen.add(grid[newRow][newCol])
                    ans = max(ans, 1+sum(area[i] for i in seen))

        return ans