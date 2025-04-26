'''
Leetcode 934: Shortest Bridge

You are given an n x n binary matrix grid where 1 represents land and 0 represents water.
An island is a 4-directionally connected group of 1's not connected to any other 1's. There are exactly two islands in grid.
You may change 0's to 1's to connect the two islands to form one island.
Return the smallest number of 0's you must flip to connect the two islands.

Example 1:
Input: grid = [[0,1],[1,0]]
Output: 1

Example 2:
Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2

Example 3:
Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1
'''
from typing import List
from collections import deque
from itertools import product

# First comment in this one is the best solution: https://leetcode.com/problems/shortest-bridge/discuss/374415/python-easy-to-understand-with-key-comments
# https://patricksudo.com/blogs/LeetCode/0934-Shortest-Bridge.html#solution
# https://leetcode.com/problems/shortest-bridge/discuss/189293/C%2B%2B-BFS-Island-Expansion-%2B-UF-Bonus
# https://leetcode.com/problems/shortest-bridge/discuss/374415/python-easy-to-understand-with-key-comments
"""
1. Use DFS to find the first island, replace all 1 with 2. Do not call DFS twice, otherwise the second island will become 2, too.
2. Next, use BFS to control the expansion of the first island and count the steps until it reaches the second island (look for number 1).
"""
class Solution:
    def shortestBridge(self, A: List[List[int]]) -> int:
        # Check if given cell (row, col) is within bounds in a nrows x ncols matrix
        def isWithinBounds(row, col):
            return 0 <= row < nrows and 0 <= col < ncols
        
        # DFS to find all connected cells for one of the islands
        def dfs(row, col):
            if not isWithinBounds(row,col) or (row, col) in seen or A[row][col] != 1:
                return
            seen.add( (row, col) )
            distance = 0
            bfsQueue.append( (row, col, distance) )
            # Mark 1st island cells with -1 to distinguish from second island's 1 values
            A[row][col] = -1
            # DFS Search in all 4 directions
            for dr, dc in dirs:
                dfs(row+dr, col+dc)

        dirs = [ [0,1], [0,-1], [1,0], [-1,0] ]
        # 1. DFS to find the first island, and mark it all with -1's
        seen, bfsQueue, nrows, ncols = set(), deque(), len(A), len(A[0])
        for row, col in product(range(nrows), range(ncols)):
            if A[row][col]:
                dfs(row, col)
                break
        
        # 2. BFS to find the second island, and update the shortest distance
        while bfsQueue:
            row, col, distance = bfsQueue.popleft()
            if A[row][col] == 1:  # Found second island
                # This must be shortest distance, because BFS searches level-by-level
                return distance-1
            # BFS Search in all 4 directions
            for dr, dc in dirs:
                rowNew = row+dr
                colNew = col+dc
                if isWithinBounds(rowNew, colNew) and (rowNew, colNew) not in seen:
                    seen.add( (rowNew, colNew) )
                    bfsQueue.append( (rowNew, colNew, distance+1) )
        return -1