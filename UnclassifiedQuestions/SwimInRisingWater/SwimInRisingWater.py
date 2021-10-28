"""
778. Swim in Rising Water
Solution 1: Heap
Solution 2: Binary Search + DFS
"""
class Solution:
    """
    Use Heap.
     - Time complexity: O(N^2 * logN). We may expand O(N^2) nodes,
       and each one requires O(logN) time to perform the heap ops.
    """
    def swimInWater(self, grid: List[List[int]]) -> int:
        n = len(grid)
        visited = { (0,0) }
        # Heap elements :(depth, row, column)
        heap = [ (grid[0][0], 0, 0) ]
        result = 0
        directions = [ [0,1], [0,-1], [1,0], [-1,0] ]

        def isWithinBounds(row: int, col: int) -> bool:
            return 0 <= row < n and 0 <= col < n

        while heap:
            depth, row, col = heapq.heappop(heap)
            result = max(result, depth)
            # Solution found: Reached (n-1, n-1)
            if row == col == n-1:
                return result
            # Check neighbours
            for dRow, dCol in directions:
                newRow, newCol = row+dRow, col+dCol
                if isWithinBounds(newRow, newCol) and (newRow, newCol) not in visited:
                    heapq.heappush(heap, (grid[newRow][newCol], newRow, newCol) )
                    visited.add( (newRow, newCol) )

    """
    Binary Search + DFS
     - Time Complexity: O(N^2 * log(N)). Our depth-first search
       during a call to possible is O(N^2), and we make up to O(logN)
       of them between [grid[0][0], n^2].
    """
    def swimInWaterBinarySearchDfs(self, grid: List[List[int]]) -> int:
        n = len(grid)
        directions = [ [0,1], [0,-1], [1,0], [-1,0] ]

        def isWithinBounds(row: int, col: int) -> bool:
            return 0 <= row < n and 0 <= col < n

        def dfsPossible(T):
            stack = [ (0,0) ]
            visited = { (0,0) }
            while stack:
                row, col = stack.pop()
                # Solution found: location (n-1, n-1)
                if row == col == n-1:
                    return True
                # Check neighbours
                for dRow, dCol in directions:
                    newRow, newCol = row+dRow, col+dCol
                    if isWithinBounds(newRow, newCol) and (newRow, newCol) not in visited and grid[newRow][newCol] <= T:
                        stack.append( (newRow,newCol) )
                        visited.add( (newRow, newCol) )
            return False

        # Main starts here
        # Find smallest T where dfs to reach (n-1, n-1) is possible
        low, high = grid[0][0], n*n
        while low < high:
            mid = (low+high)//2
            if not dfsPossible(mid):
                low = mid+1
            else:
                high = mid
        return low