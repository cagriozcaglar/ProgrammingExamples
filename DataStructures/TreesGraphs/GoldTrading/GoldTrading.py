"""
Asked by Snap
"""
"""
Gold trading

A B C D E
F G H I J
K L M N O
P Q R S T

1. initially there's just one country has gold
2. a country with gold can sell to neighbors (u, d, l, r)
3. To buy gold, the neighbor has to have GDP >= T
4. GDP inc by 1 each year for all countries

Q: Given a target country, how many years before it gets gold?

Example:
GDP
2. 3. 1. 5. 8 
2. 2. 5. 6. 7
1. 1. 1. 3. 2
20 5. 5. 5. 5

src_country: (1, 1) -> G
dst_country: (3, 0) -> P
T: 6

Ans: 3

Year 0: G
Year 1: G, H, I, D, J, E
Year 2: ...
Year 3: 
"""

"""
2. 3. 1. 5. 8 
2. 2. 5. 6. 7
1. 1. 1. 3. 2
20 5. 5. 5. 5

A B C D E
F G H I J
K L M N O
P Q R S T
"""

"""
gold, threshold
"""
"""
BFS
1. Search neighbours
2. Increase GDPs of all by 1 every depth
"""
from collections import deque
from typing import List

def findNumYearsUntilTarget(grid: List[List[int]], src: List[int], dest: List[int], threshold: int) -> int:
    nrows, ncols = len(grid), len(grid[0])
    dirs = [ [0,1], [0, -1], [-1, 0], [1,0] ]

    def isWithinBounds(row, col) -> bool:
        return 0 <= row < nrows and 0 <= col < ncols

    def incrementGdp() -> None:
        grid = [grid[row][col]+1 for row in range(nrows) for col in range(ncols)]

    # Initial params
    year = 0
    hasGold = set(src)
    bfsQueue: deque([ (src, 0) ])

    while True:
        while bfsQueue:
            point, depth = bfsQueue.popleft()
            # Success
            if point == dest:
                return year
            # Else check neighbours
            row, col = point
            for dRow, dCol in dirs:
                newRow = row+dRow
                newCol = row+dCol
                newPoint = [newRow, newCol]
                if newPoint not in hasGold and isWithinBounds(newRow, newCol) and grid[row][col] >= threshold:
                    bfsQueue.append(newPoint)
        # Next year
        incrementGdp(grid)
        year += 1

    return year









