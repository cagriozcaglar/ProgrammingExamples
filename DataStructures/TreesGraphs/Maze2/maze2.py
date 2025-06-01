'''
Leetcode 505: Maze 2

There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] and destination = [destinationrow, destinationcol], return the shortest distance for the ball to stop at the destination. If the ball cannot stop at destination, return -1.

The distance is the number of empty spaces traveled by the ball from the start position (excluded) to the destination (included).

You may assume that the borders of the maze are all walls (see examples).
'''
from typing import List
import math
from collections import deque

'''
# Solve with BFS.
    - Time complexity : O(m * n * max(m,n)). Complete traversal of maze will be done in the worst case. Here, m and n refers to the number of rows and columns of the maze. Further, for every current node chosen, we can travel upto a maximum depth of max(m,n) in any direction.
    - Space complexity : O(mn). queue size can grow upto m*n in the worst case.
'''
class Solution:
    def shortestDistance(self, maze: List[List[int]], start: List[int], destination: List[int]) -> int:
        # Initialize variables
        m, n = len(maze), len(maze[0])
        # visited array not needed. We may need to revisit repeatedly
        dists = [[math.inf] * n for _ in range(m)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # Destination point
        dest_r, dest_c = destination

        # within_bounds method
        def within_bounds(row: int, col: int):
            return 0 <= row < m and 0 <= col < n

        # BFS
        queue = deque([start])
        dists[start[0]][start[1]] = 0  # Starting point distance is 0

        while queue:
            i, j = queue.popleft()

            for d_r, d_c in directions:
                # IMPORTANT: Reset starting cell's x,y inside the for loop
                r, c, current_dist = i, j, dists[i][j]

                # Keep going in the same direction until hitting a wall or bounds
                while within_bounds(r + d_r, c + d_c) and maze[r + d_r][c + d_c] == 0:
                    r, c = r + d_r, c + d_c
                    current_dist += 1

                # Update minimum distance
                if current_dist < dists[r][c]:
                    dists[r][c] = current_dist
                    # Add new position to the queue for further exploration
                    queue.append((r,c))

        return -1 if dists[dest_r][dest_c] == math.inf else dists[dest_r][dest_c]