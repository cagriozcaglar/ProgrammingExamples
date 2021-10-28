# 787. Cheapest Flights Within K Stops
import heapq
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
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