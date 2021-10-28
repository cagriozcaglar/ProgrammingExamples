"""
1168. Optimize Water Distribution in a Village
There are n houses in a village. We want to supply water for all the houses by building wells and laying pipes.
For each house i, we can either build a well inside it directly with cost wells[i - 1] (note the -1 due to 0-indexing),
or pipe in water from another well to it. The costs to lay pipes between houses are given by the array pipes, where each
pipes[j] = [house1j, house2j, costj] represents the cost to connect house1j and house2j together using a pipe.
Connections are bidirectional.
Return the minimum total cost to supply water to all houses.
"""

class Solution:
    """
    Prim's MST algorithm:
     - N: number of houses
     - M: number of pipes
     - Time Complexity: O((N+M) * log(N+M) )
      - Build the graph: O((N+M))
      - Build the MST: O((N+M) * log(N+M) ) (Iterate through all edges (O((N+M))), and pop + push to heap for entry /
        exit to / from heap: O(log(N+M)))
    """
    def minCostToSupplyWaterPrim(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        # Adjacency list
        graph = defaultdict(list)
        # Add virtual node for wells
        for index, cost in enumerate(wells):
            graph[0].append((cost,index+1))
        # Add edges from pipes
        for house1, house2, cost in pipes:
            graph[house1].append((cost, house2))
            graph[house2].append((cost, house1))

        # Set to maintain all vertices added to MST, add vertex 0
        mstSet = set([0])
        # Heap to maintain order of edges to visit
        # Arbitrary choice: start from vertex 0.
        heapq.heapify(graph[0])
        edgesHeap = graph[0]

        totalCost = 0
        while len(mstSet) < n+1:
            cost, nextHouse = heapq.heappop(edgesHeap)
            if nextHouse not in mstSet:
                # Add new vertex into set
                mstSet.add(nextHouse)
                totalCost += cost
                # Check neighbours of nextHouse
                for newCost, neighHouse in graph[nextHouse]:
                    if neighHouse not in mstSet:
                        heapq.heappush(edgesHeap, (newCost, neighHouse))

        return totalCost
    """
    Kruskal's MST algorithm
     - N: number of houses
     - M: number of pipes
     - Time Complexity: O((N+M) * log(N+M) )
      - Build the graph: O((N+M))
      - Sort the edges: O((N+M) * log(N+M) )
      - Iterate over edges, invoke union-find operation for each: O((N+M) * log(N) )
      - Build the MST: O((N+M) * log(N+M) )
    """
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        orderedEdges = []
        # Add virtual vertex and edges for wells
        for index, weight in enumerate(wells):
            orderedEdges.append( (weight, 0, index+1) )
        # Add existing edges from pipes
        for house1, house2, weight in pipes:
            orderedEdges.append( (weight, house1, house2) )

        # Sort edges by their increasing weights
        orderedEdges.sort(key = lambda x: x[0])

        # Iterate through ordered edges
        uf = UnionFind(n)
        totalCost = 0
        for cost, house1, house2 in orderedEdges:
            # Check if we should add the new edge
            # if union returns in union of diff groups
            if uf.union(house1, house2):
                totalCost += cost

        return totalCost

    class UnionFind:
        def __init__(self, size):
            self.group = [i for i in range(size+1)]
            self.rank = [0] * (size+1)

    def find(self, person: int) -> int:
        if self.group[person] != person:
            self.group[person] = self.find(self.group[person])
        return self.group[person]

    def union(self, person1: int, person2: int) -> bool:
        group1 = self.find(person1)
        group2 = self.find(person2)
        if group1 == group2:
            return False

        # Attach group of lower rank to group with highes rank
        if self.rank[group1] > self.rank[group2]:
            self.group[group2] = group1
        elif self.rank[group1] < self.rank[group2]:
            self.group[group1] = group2
        else:
            self.group[group1] = group2
            self.group[group2] += 1
        return True
