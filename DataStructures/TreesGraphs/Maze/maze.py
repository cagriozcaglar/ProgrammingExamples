'''
Leetcode 490: The Maze

There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.

Given the ball's start position, the destination and the maze, determine whether the ball could stop at the destination.

The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume that the borders of the maze are all walls. The start and destination coordinates are represented by row and column indexes.

Example 1:
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)

Output: true

Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.
'''
from typing import List
from collections import deque

class Solution:
    def hasPath(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
        # Initialize variables
        m, n = len(maze), len(maze[0])
        visited = [[False] * n for _ in range(m)]
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # within_bounds method
        def within_bounds(row: int, col: int):
            return 0 <= row < m and 0 <= col < n

        # BFS
        queue = deque([start])
        visited[start[0]][start[1]] = True

        while queue:
            cell = queue.popleft()
            if cell == destination:
                return True

            for direction in directions:
                # IMPORTANT: Reset starting cell's x,y inside the for loop
                r, c = cell
                d_r, d_c = direction[0], direction[1]

                # Keep going in the same direction until hitting a wall or bounds
                while within_bounds(r + d_r, c + d_c) and maze[r + d_r][c + d_c] == 0:
                    r, c = r + d_r, c + d_c

                # Update visited
                if not visited[r][c]:
                    queue.append([r, c])
                    visited[r][c] = True
        return False