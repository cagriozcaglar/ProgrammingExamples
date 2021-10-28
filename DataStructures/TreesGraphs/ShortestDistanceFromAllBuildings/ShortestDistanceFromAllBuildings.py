"""
317. Shortest Distance from All Buildings
You are given an m x n grid grid of values 0, 1, or 2, where:
- each 0 marks an empty land that you can pass by freely,
- each 1 marks a building that you cannot pass through, and
- each 2 marks an obstacle that you cannot pass through.
You want to build a house on an empty land that reaches all buildings in the shortest total travel distance. You can
only move up, down, left, and right.
Return the shortest travel distance for such a house. If it is not possible to build such a house according to the
above rules, return -1.
The total travel distance is the sum of the distances between the houses of the friends and the meeting point.
The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
"""
# https://leetcode.com/problems/shortest-distance-from-all-buildings/discuss/76877/Python-solution-72ms-beats-100-BFS-with-pruning
class Solution:
    def shortestDistance(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return -1
        m, n = len(grid), len(grid[0])
        numBuildings = sum(val for line in grid for val in line if val==1)
        numReach, distSum = [[0] * n for _ in range(m)], [[0] * n for _ in range(m)]
        directions = [ [0, 1], [0, -1], [1, 0], [-1, 0] ]

        def isWithinBounds(row, col):
            return 0 <= row < m and 0 <= col < n

        def bfs(row, col):
            visited = [ [False]*n for _ in range(m) ]
            visited[row][col], count1 = True, 1
            bfsQueue = deque([(row, col, 0)])
            while bfsQueue:
                x, y, distance = bfsQueue.popleft()
                for dRow, dCol in directions:
                    newRow, newCol = x+dRow, y+dCol
                    if isWithinBounds(newRow, newCol) and not visited[newRow][newCol]:
                        visited[newRow][newCol] = True
                        if grid[newRow][newCol] == 0:
                            bfsQueue.append( (newRow, newCol, distance+1) )
                            numReach[newRow][newCol] += 1
                            distSum[newRow][newCol] += distance+1
                        elif grid[newRow][newCol] == 1:
                            count1 += 1
            return count1 == numBuildings

        for row in range(m):
            for col in range(n):
                if grid[row][col] == 1:
                    if not bfs(row, col):
                        return -1
        # Calculate minimum distance
        #return min([distSum[i][j] for i in range(m) for j in range(n) if not grid[i][j] and numReach[i][j] == numBuildings] or [-1])

        distances = []
        for row in range(m):
            for col in range(n):
                if not grid[row][col] and numReach[row][col] == numBuildings:
                    distances.append(distSum[row][col])
        return min(distances or [-1])