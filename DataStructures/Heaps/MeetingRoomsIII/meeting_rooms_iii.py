'''
Leetcode 2402: Meeting Rooms III

You are given an integer n. There are n rooms numbered from 0 to n - 1.

You are given a 2D integer array meetings where meetings[i] = [starti, endi] means that a meeting will be held during the half-closed time interval [starti, endi). All the values of starti are unique.

Meetings are allocated to rooms in the following manner:

1) Each meeting will take place in the unused room with the lowest number.
2) If there are no available rooms, the meeting will be delayed until a room becomes free. The delayed meeting should have the same duration as the original meeting.
3) When a room becomes unused, meetings that have an earlier original start time should be given the room.

Return the number of the room that held the most meetings. If there are multiple rooms, return the room with the lowest number.

A half-closed interval [a, b) is the interval between a and b including a and not including b.

Example 1:
Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
Output: 0
Explanation:
- At time 0, both rooms are not being used. The first meeting starts in room 0.
- At time 1, only room 1 is not being used. The second meeting starts in room 1.
- At time 2, both rooms are being used. The third meeting is delayed.
- At time 3, both rooms are being used. The fourth meeting is delayed.
- At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
- At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
Both rooms 0 and 1 held 2 meetings, so we return 0. 
'''

from typing import List, Dict
import heapq

class Solution:
    '''
    Algorithm:
    0) ** While ** there are used rooms in meeting_end_time_min_heap and the first room's
        meeting has already concluded (meeting end time <= current meeting start time),
        remove the room from meeting_end_time_min_heap and add it back to available_room_index_min_heap.
    1) If there is room left in available_room_index_min_heap, use the one with min index:
        - Extract room_index from available_room_index_min_heap, Push (end, room_index) to meeting_end_time_min_heap
    2) If no room left, delay the meeting.
        - Pop meeting from meeting_end_time_min_heap with (end, room_index).
        - room_index becomes free.
        - Allocate delayed_meeting as follows: Push (end = end_prev + start_new - end_new, room_index) to meeting_end_time_min_heap.
        - available_room_index_min_heap doesn't change, because room is occupied the whole time
    '''
    def mostBooked(self, n: int, meetings: List[List[int]]) -> int:
        # Room index min-heap: Used to find the room with min index
        available_room_index_min_heap = list(range(n))
        # Build room_index_min_heap
        heapq.heapify(available_room_index_min_heap)

        # Meeting end time min-heap: Used to find the meeting ending soonest
        # Will later push element: (end, room_index)
        meeting_end_time_min_heap = []

        # Hash table: Used to map room indices to number of meetings hold
        room_meeting_count_map: Dict[int, int] = defaultdict(int)

        # Iterate over meetings:
        # Sort meetings by start time, because of rule 3: When a room becomes unused,
        # meetings that have an earlier original start time should be given the room.
        for [start, end] in sorted(meetings):
            # 0) ** While ** there are used rooms in meeting_end_time_min_heap and the first room's
            # meeting has already concluded (meeting end time <= current meeting start time),
            # remove the room from meeting_end_time_min_heap and add it back to available_room_index_min_heap.
            while meeting_end_time_min_heap and meeting_end_time_min_heap[0][0] <= start:
                end_time, room_index = heapq.heappop(meeting_end_time_min_heap)
                heapq.heappush(available_room_index_min_heap, room_index)

            # 1) If there is room left in available_room_index_min_heap, use the one with min index:
            if available_room_index_min_heap:
                room_index = heapq.heappop(available_room_index_min_heap)
                heapq.heappush(meeting_end_time_min_heap, (end, room_index))
            else:
                # 2) If no room left, delay the meeting.
                #  - Pop meeting from meeting_end_time_min_heap with (end, start, room_index).
                #  - room_index becomes free, but then added to new meeting, increment count for room_index
                #  - Allocate delayed_meeting: Push (end_prev+end_new-start_new, room_index) to meeting_end_time_min_heap.
                #  - available_room_index_min_heap doesn't change, because room is occupied the whole time
                (end_prev, room_index) = heapq.heappop(meeting_end_time_min_heap)
                heapq.heappush(meeting_end_time_min_heap, (end_prev + (end-start), room_index))

            room_meeting_count_map[room_index] += 1

        # Find room with max count, then return the one with lowest index
        max_count = max(room_meeting_count_map.values())
        return min([room_index for room_index, count in room_meeting_count_map.items() if count == max_count])