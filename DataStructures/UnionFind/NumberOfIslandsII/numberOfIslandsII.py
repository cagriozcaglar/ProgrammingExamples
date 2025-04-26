"""
Number of Islands II
"""
from typing import List

class UnionFind:
    def __init__(self, n: int):
        self.count = 0
        self.parent = [-1] * n
        self.rank = [0] * n

    def isValid(self, i: int) -> bool:
        return self.parent[i] >= 0

    def setParent(self, i: int) -> None:
        if self.parent[i] == -1:
            self.parent[i] = i
            self.count += 1

    # Find with path compression
    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    # Union by rank
    def union(self, x: int, y: int) -> None:
        rootx: int = self.find(x)
        rooty: int = self.find(y)
        if(rootx != rooty):
            if self.rank[rootx] > self.rank[rooty]:
                self.parent[rooty] = rootx
            elif self.rank[rootx] < self.rank[rooty]:
                self.parent[rootx] = rooty
            else:
                self.parent[rooty] = rootx
                self.rank[rootx] += 1
            self.count -= 1

    def getCount(self) -> int:
        return self.count

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        ans = []
        directions = [ [-1,0], [1,0], [0, -1], [0, 1] ]
        uf: UnionFind = UnionFind(m*n)

        for pos in positions:
            r, c = pos[0], pos[1]
            overlap = []
            for dx, dy in directions:
                newRow = r+dx
                newCol = c+dy
                cell = (newRow * n) + newCol
                if m > newRow >= 0 and n > newCol >= 0 and uf.isValid(cell):
                    overlap.append(cell)

            index : int = r*n+c
            uf.setParent(index)
            for i in overlap:
                uf.union(i, index)
            ans.append(uf.getCount())

        return ans


"""
class UnionFind:
    def __init__(self, grid: List[List[int]]):
        self.count = 0
        m = len(grid)
        n = len(grid[0])
        self.parent = [-1] * n*m
        self.rank = [0] * n*m
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    parent[i*n+j] = i*n+j
                    count += 1
"""