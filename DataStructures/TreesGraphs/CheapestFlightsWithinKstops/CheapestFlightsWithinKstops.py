'''
Leetcodd 787: Cheapest Flights Within K Stops

There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.
You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

Example 1:
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation: The graph is shown.

Example 2:
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0
Output: 500
Explanation: The graph is shown.
'''
# Solution from https://guides.codepath.org/compsci/Cheapest-Flights-Within-K-Stops
# Uses Dijkstra's shortest-path algorithm
import heapq
from typing import List, Tuple
import math

class Solution:
    def findCheapestPrice1(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # Initialize graph adj_list
        graph = [[] for _ in range(n)]
        # Populate graph: src -> [(dest, weight),...]
        for u, v, w in flights:
            graph[u].append((v, w))

        # Return min distance within k stops
        return self.dijkstra(graph, src, dst, k)

    def dijkstra(
        self,
        graph: List[List[Tuple[int, int]]],
        src: int,
        dst: int,
        k: int
    ) -> int:
        dist = [[math.inf] * (k + 2) for _ in range(len(graph))]
        dist[src][k + 1] = 0

        # Min heap has (d, u, stops)
        min_heap = [(dist[src][k + 1], src, k + 1)]

        while min_heap:
            d, u, stops = heapq.heappop(min_heap)
            # Destination found, return distance d
            if u == dst:
                return d
            # If k stops reached, continue
            if stops == 0 or d > dist[u][stops]:
                continue
            # Check neighbours
            for v, w in graph[u]:
                if d + w < dist[v][stops - 1]:
                    dist[v][stops - 1] = d + w
                    heapq.heappush(min_heap, (dist[v][stops - 1], v, stops - 1))

        return -1

    def findCheapestPrice2(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        # Build adj matrix
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for sr, dest, distance in flights:
            adj_matrix[sr][dest] = distance

        # Shortest distances / current stops arrays for each vertex
        # Both set to max value by default
        distances = [float("inf") for _ in range(n)]
        current_stops = [float("inf") for _ in range(n)]
        distances[src], current_stops[src] = 0, 0

        # Each Node in heap holds: (cost (key), stops, node)
        minHeap = [(0, 0, src)]
        while minHeap:
            cost, stops, node = heapq.heappop(minHeap)
            # If dest is reached, return the cost
            if node == dst:
                return cost
            # If no more steps left, continue
            if stops == k+1:
                continue
            # Examine & relax all neighbour edges if possible
            for neigh in range(n):
                if adj_matrix[node][neigh] > 0:
                    dU, dV, wUV = cost, distances[neigh], adj_matrix[node][neigh]
                    # Better cost?
                    if dU + wUV < dV:
                        distances[neigh] = dU + wUV
                        heapq.heappush(minHeap, (dU+wUV, stops+1, neigh))
                    # Less stops?
                    elif stops < current_stops[neigh]:
                        heapq.heappush(minHeap, (dU+wUV, stops+1, neigh))
                    current_stops[neigh] = stops

        return -1 if distances[dst] == float("inf") else distances[dst]