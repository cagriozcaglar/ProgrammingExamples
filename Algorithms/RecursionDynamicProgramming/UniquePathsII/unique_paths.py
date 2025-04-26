'''
Leetcode 63: Unique Paths II

You are given an m x n integer array grid. There is a robot initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.
An obstacle and space are marked as 1 or 0 respectively in grid. A path that the robot takes cannot include any square that is an obstacle.
Return the number of possible unique paths that the robot can take to reach the bottom-right corner.

Example 1:
Input: obstacleGrid = [[0,0,0],[0,1,0],[0,0,0]]
Output: 2
Explanation: There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right

Example 2:
Input: obstacleGrid = [[0,1],[0,0]]
Output: 1
'''
from typing import List

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        # If starting cell has obstacle, return 0, no path to destination
        if obstacleGrid[0][0] == 1:
            return 0

        rows, cols = len(obstacleGrid), len(obstacleGrid[0])
        ways = [[0] * cols for _ in range(rows)]

        # Number of ways to reach starting cell = 1 (we know obstacleGrid[0][0] == 0)
        ways[0][0] = 1

        # Fill values for first column
        for i in range(1, rows):
            # If there is no obstacle, use the value from the cell above
            # Otherwise, ways[i][0] = 0, which is default value
            if obstacleGrid[i][0] == 0:
                ways[i][0] = ways[i-1][0]  

        # Fill values for first row
        for j in range(1, cols):
            # If there is no obstacle, use the value from the cell to the left
            # Otherwise, ways[0][j] = 0, which is default value
            if obstacleGrid[0][j] == 0:
                ways[0][j] = ways[0][j-1]

        # Iterate over all other cells and update ways array
        # Start from cell (1,1), fill the values
        # Num of ways: ways[i][j] = ways[i-1][j] + cell[i][j-1] (from above and left)
        for i in range(1, rows):
            for j in range(1, cols):
                # If there is no obstacle, use sum of left and above cells
                # Otherwise, ways[0][j] = 0, which is default value
                if obstacleGrid[i][j] == 0:
                    ways[i][j] = ways[i-1][j] + ways[i][j-1]

        # Return value stored in rightmost bottommost cell.
        # return ways[rows-1][cols-1]
        return ways[-1][-1]
