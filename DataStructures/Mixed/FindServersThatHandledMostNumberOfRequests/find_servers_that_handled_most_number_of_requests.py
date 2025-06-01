'''
Leetcode 1606: Find Servers That Handled Most Number of Requests

You have k servers numbered from 0 to k-1 that are being used to handle multiple requests simultaneously. Each server has infinite computational capacity but cannot handle more than one request at a time. The requests are assigned to servers according to a specific algorithm:

The ith (0-indexed) request arrives.
If all servers are busy, the request is dropped (not handled at all).
If the (i % k)th server is available, assign the request to that server.
Otherwise, assign the request to the next available server (wrapping around the list of servers and starting from 0 if necessary). For example, if the ith server is busy, try to assign the request to the (i+1)th server, then the (i+2)th server, and so on.
You are given a strictly increasing array arrival of positive integers, where arrival[i] represents the arrival time of the ith request, and another array load, where load[i] represents the load of the ith request (the time it takes to complete). Your goal is to find the busiest server(s). A server is considered busiest if it handled the most number of requests successfully among all the servers.

Return a list containing the IDs (0-indexed) of the busiest server(s). You may return the IDs in any order.
'''
from typing import List
from collections import heapq
from sortedcontainers import SortedList

class Solution:
    def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
        # server loads
        count = [0] * k

        # Data structures: SortedList for "free" servers, min-heap for "busy" servers
        # free: SortedList of free server indices
        # busy: Min-heap with key server job end time => Push (end_time, server_id)
        # All servers are free at the beginning
        busy, free = [], SortedList(list(range(k)))

        for i, start in enumerate(arrival):
            # Move free servers from busy to free
            while busy and busy[0][0] <= start:
                end_time, server_id = heapq.heappop(busy)
                free.add(server_id)  # takes care of ordering in O(log(k))

            # If we have free servers, use binary search to find the target server
            if free:
                index = free.bisect_left(i % k)
                # Bisect left search returns left index. if index >= len(free), return free[0]
                server_id = free[index] if index < len(free) else free[0]
                free.remove(server_id)
                heapq.heappush(busy, (start + load[i], server_id))
                count[server_id] += 1

        # Find servers that have maximum workload
        max_job = max(count)
        return [i for i, n in enumerate(count) if n == max_job]