'''
Leetcode 815: Bus Routes

You are given an array routes representing bus routes where routes[i] is a bus route that the ith bus repeats forever.

For example, if routes[0] = [1, 5, 7], this means that the 0th bus travels in the sequence 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... forever.

You will start at the bus stop source (You are not on any bus initially), and you want to go to the bus stop target. You can travel between bus stops by buses only.

Return the least number of buses you must take to travel from source to target. Return -1 if it is impossible.

Example 1:
Input: routes = [[1,2,7],[3,6,7]], source = 1, target = 6
Output: 2
Explanation: The best strategy is take the first bus to the bus stop 7, then take the second bus to the bus stop 6.

Example 2:
Input: routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12
Output: -1
'''
from typing import List, Dict
from collections import deque, defaultdict

class Solution:
    # Similar to https://github.com/hogan-tech/leetcode-solution/blob/main/Python/0815-bus-routes.py
    def numBusesToDestinationV1(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0

        bus_stop_to_route_indices: Dict[int, List[int]] = defaultdict(list)

        # Create map from the bus stop to all routes that include this stop.
        for route_index, route in enumerate(routes):
            for stop in route:
                bus_stop_to_route_indices[stop].append(route_index)

        # BFS
        # Queue element: tuple of current bus stop and buses taken till then
        bfs_queue = deque([(source, 0)])
        visited_stops = set()

        while bfs_queue:
            current_stop, buses = bfs_queue.popleft()
            if current_stop == target:
                return buses
            for route_index in bus_stop_to_route_indices[current_stop]:
                for bus_stop in routes[route_index]:
                    if bus_stop not in visited_stops:
                        visited_stops.add(bus_stop)
                        bfs_queue.append( (bus_stop, buses + 1) )
                routes[route_index] = []

        return -1

    # Similar to https://medium.com/@aruvanshn/crack-the-code-master-the-bus-routes-leetcode-challenge-bcf2879b1d80 
    def numBusesToDestinationV2(self, routes: List[List[int]], source: int, target: int) -> int:
        if source == target:
            return 0

        bus_stop_to_route_indices: Dict[int, List[int]] = defaultdict(list)
        seen_route = set()
        seen_node = set()

        # Create map from the bus stop to all routes that include this stop.
        for route_index, route in enumerate(routes):
            for stop in route:
                bus_stop_to_route_indices[stop].append(route_index)

        # BFS
        # Queue element: tuple of current bus stop and buses taken till then
        bfs_queue = deque([(source, 0)])

        while bfs_queue:
            current_stop, buses = bfs_queue.popleft()
            if current_stop == target:
                return buses
            for route_index in bus_stop_to_route_indices[current_stop]:
                if route_index in seen_route:
                    continue
                seen_route.add(route_index)
                for node in routes[route_index]:
                    if node in seen_node:
                        continue
                    seen_node.add(node)
                    bfs_queue.append( (node, buses + 1) )

        return -1