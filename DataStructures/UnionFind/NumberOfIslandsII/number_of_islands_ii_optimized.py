'''
Leetcode 305: Number of Islands II

You are given an empty 2D binary grid grid of size m x n. The grid represents a map where 0's represent water and 1's represent
land. Initially, all the cells of grid are water cells (i.e., all the cells are 0's).

We may perform an add land operation which turns the water at position into a land. You are given an array positions where
positions[i] = [ri, ci] is the position (ri, ci) at which we should operate the ith operation.

Return an array of integers answer where answer[i] is the number of islands after turning the cell (ri, ci) into a land.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all
four edges of the grid are all surrounded by water.
'''
from typing import List

class UnionFind:
    def __init__(self, size: int):
        self.parent = [-1] * size
        self.rank = [0] * size
        self.count = 0

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        x_root = self.find(x)
        y_root = self.find(y)
        if x_root == y_root:
            return
        elif self.rank[x_root] < self.rank[y_root]:
            self.parent[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.parent[y_root] = x_root
        else:
            self.parent[y_root] = x_root
            self.rank[x_root] += 1
        self.count -= 1  # Reduce number of islands after union

    def add_land(self, x: int) -> None:
        # If not land, add land and increase count
        if not self.is_land(x):
            self.parent[x] = x
            self.count += 1

    def is_land(self, x: int) -> bool:
        return self.parent[x] >= 0

    def number_of_islands(self):
        return self.count

class Solution:
    def numIslands2(self, m: int, n: int, positions: List[List[int]]) -> List[int]:
        x = [-1, 1, 0, 0]
        y = [0, 0, -1, 1]

        union_find: UnionFind = UnionFind(m * n)
        answer: List[int] = []

        for position in positions:
            land_position = position[0] * n + position[1]
            union_find.add_land(land_position)

            # Check neighbours, update connected components using union
            for i in range(len(x)):
                neighbourX = position[0] + x[i]
                neighbourY = position[1] + y[i]
                neighbourPosition = neighbourX * n + neighbourY
                # If there is land in neighbourPosition, union
                if (0 <= neighbourX < m) and \
                   (0 <= neighbourY < n) and \
                   union_find.is_land(neighbourPosition):
                    union_find.union(land_position, neighbourPosition)
            # Return num_islands after this position
            answer.append(union_find.number_of_islands())

        return answer
