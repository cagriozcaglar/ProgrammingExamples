'''
Leetcode 286: Walls and Gates

You are given an m x n grid rooms initialized with these three possible values.

-1 A wall or an obstacle.
0 A gate.
INF Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

Example 1:
Input: rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,
2147483647,2147483647]]
Output: [[3,-1,0,1],[2,2,1,-1],[1,-1,2,-1],[0,-1,3,4]]

Example 2:
Input: rooms = [[-1]]
Output: [[-1]]
'''

'''
Solution: BFS from gates to empty rooms
 - Instead of searching from an empty room to the gates, how about searching the other way round?
   In other words, we initiate breadth-first search (BFS) from all gates at the same time. Since BFS
   guarantees that we search all rooms of distance d before searching rooms of distance d + 1, the
   distance to an empty room must be the shortest.
 - Note: Does not return anything, modifies rooms in-place instead.
 - Time Complexity: O(m*n)
 - Space Complexity: O(m*n)
'''
from enum import Enum
from typing import List, Tuple
from collections import deque

class Cell(Enum):
    EMPTY = 2147483647
    GATE = 0
    WALL = -1

class Solution:
    def wallsAndGates(self, rooms: List[List[int]]) -> None:
        n_rows, n_cols = len(rooms), len(rooms[0])

        # Four directions
        directions = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ]

        def is_within_bounds(row_index: int, col_index: int) -> bool:
            return 0 <= row_index < n_rows and 0 <= col_index < n_cols

        # BFS from gates to empty rooms. Each entry is a 2-tuple representing (row, col)
        bfs_queue: List[Tuple(int, int)] = deque([])
        # Iterate over all cells, add all gates to initial BFS queue
        # This is a multi-source BFS queue. When we reach an empty from any
        # of the gates, the distance for this empty room is guaranteed to be
        # the shortest distance from a gate to the empty room.
        for row in range(n_rows):
            for col in range(n_cols):
                if rooms[row][col] == Cell.GATE.value:
                    bfs_queue.append((row, col))

        # Iterate on BFS queue until it is empty
        while bfs_queue:
            # IMPORTANT: Left side has parentheses around it, showing it is a tuple
            # If you don't do this, it will return an error
            row, col = bfs_queue.popleft()
            for dir_x_y in directions:
                new_row, new_col = row + dir_x_y[0], col + dir_x_y[1]
                if is_within_bounds(new_row, new_col) and \
                   rooms[new_row][new_col] == Cell.EMPTY.value:
                   rooms[new_row][new_col] = rooms[row][col] + 1
                   bfs_queue.append((new_row, new_col))