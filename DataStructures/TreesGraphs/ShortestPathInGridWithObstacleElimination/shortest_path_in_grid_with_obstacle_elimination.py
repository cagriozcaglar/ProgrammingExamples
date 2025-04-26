'''
Leetcode 1293: Shortest Path in a Grid with Obstacles Elimination

You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle). You can move up, down, left, or right from and to an empty cell in one step.

Return the minimum number of steps to walk from the upper left corner (0, 0) to the lower right corner (m - 1, n - 1) given that you can eliminate at most k obstacles. If it is impossible to find such walk return -1.

Example 1:
Input: grid = [[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], k = 1
Output: 6
Explanation:
The shortest path without eliminating any obstacle is 10.
The shortest path with one obstacle elimination at position (3,2) is 6. Such path is (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (4,2).

Example 2:
Input: grid = [[0,1,1],[1,1,1],[1,0,0]], k = 1
Output: -1
Explanation: We need to eliminate at least two obstacles to find such a walk.
'''
from collections import deque
from typing import List

class Solution:
    def shortestPath(self, grid: List[List[int]], k: int) -> int:
        m, n = len(grid), len(grid[0])
        dest = (m - 1, n - 1)
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

        # if we have sufficient quotas to eliminate the obstacles in the worst case,
        # then the shortest distance is the Manhattan distance
        if k >= m + n - 2:
            return m + n - 2

        def is_within_bounds(row: int, col: int) -> bool:
            return 0 <= row < m and 0 <= col < n

        # BFS starts here
        # Queue has this in it: (num_steps, new_state), where new_state = (row, col, k)
        state = (0, 0, k)
        queue = deque([(0, state)])
        visited = set([state])

        while queue:
            num_steps, (row, col, curr_k) = queue.popleft()
            # Return if dest is reached. This is also guaranteed to be minimum num_steps.
            # As soon as the sound wave reach the object, the path is
            # guaranteed to be the shortest, since the distance is proportional
            # to the time, the more time it takes, the longer the distance is.
            if (row, col) == dest:
                return num_steps

            # Check neighbours
            for direction in directions:
                d_r, d_c = direction
                # Update row, col in direction
                new_row, new_col = row + d_r, col + d_c
                if is_within_bounds(new_row, new_col):
                    new_k = curr_k - grid[new_row][new_col]
                    new_state = (new_row, new_col, new_k)
                    # Add next move in queue if it qualifies
                    if new_k >= 0 and new_state not in visited:
                        visited.add(new_state)
                        queue.append([num_steps + 1, new_state])
        # Didn't reach destination
        return -1