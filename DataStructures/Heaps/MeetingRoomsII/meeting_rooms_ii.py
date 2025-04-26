'''
Leetcode 253: Meeting Rooms II

Given an array of meeting time intervals intervals where intervals[i] = [starti, endi], return the minimum number of conference rooms required.

Example 1:
Input: intervals = [[0,30],[5,10],[15,20]]
Output: 2

Example 2:
Input: intervals = [[7,10],[2,4]]
Output: 1
'''
from typing import List
import heapq

class Solution:
    # Solution 1: Greedy + Heap
    # Time: O(n*log(n))
    # Space: O(n)
    def minMeetingRooms_GreedyHeap(self, intervals: List[List[int]]) -> int:

        # If no meetings, no rooms allocated
        if not intervals:
            return 0

        # Min-heap Init
        free_rooms_heap = []

        # max_events_count, current_event_count = 0, 0

        # 1. Sort intervals by starting time, so we know how many events
        intervals.sort(key=lambda x: x[0])

        # Add the first meeting, add its finish time
        heapq.heappush(free_rooms_heap, intervals[0][1])

        # 2. Iterate over events.
        for start, end in intervals[1:]:
            # If room due to free up earliest (lowest finish time) is free, pop it from heap
            if free_rooms_heap[0] <= start:
                heapq.heappop(free_rooms_heap)

            # If a new room is to be assigned, add to heap
            heapq.heappush(free_rooms_heap, end)

        # Return size of heap
        return len(free_rooms_heap)

    # Solution 2: Simple array processing with chronological order
    # Time: O(n*log(n))
    # Space: O(n)
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        used_rooms = 0

        # 1. Separate out the start times and the end times in their separate arrays, and sort
        start_times = sorted([interval[0] for interval in intervals])
        end_times = sorted([interval[1] for interval in intervals])
        L = len(intervals)

        # Pointers to start_times and end_times
        s, e = 0, 0

        # Until all meetings have been processed
        while s < L:
            # If a meeting has ended before new event started
            if start_times[s] >= end_times[e]:
                # Free up a room and increment e
                used_rooms -= 1
                e += 1

            # Do this for every case.
            # If a room got free, increment used_rooms wouldn't have any effect
            # If no room is free, this will increase used_rooms
            used_rooms += 1
            s += 1

        return used_rooms