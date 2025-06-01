'''
Leetcode 759: Employee Free Time

We are given a list schedule of employees, which represents the working time for each employee.
Each employee has a list of non-overlapping Intervals, and these intervals are in sorted order.
Return the list of finite intervals representing common, positive-length free time for all employees, also in sorted order.
(Even though we are representing Intervals in the form [x, y], the objects inside are Intervals, not lists or arrays. For example, schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined).  Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length.

Example 1:
Input: schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]]
Output: [[3,4]]
Explanation: There are a total of three employees, and all common
free time intervals would be [-inf, 1], [3, 4], [10, inf].
We discard any intervals that contain inf as they aren't finite.

Example 2:
Input: schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]
Output: [[5,6],[7,9]]
'''

# Definition for an Interval.
class Interval:
    def __init__(self, start: int = None, end: int = None):
        self.start = start
        self.end = end

from typing import List

class Solution:
    '''
    Approach: Events (Line Sweep)
     - Time Complexity: O(ClogC), where C is the number of intervals across all employees.
     - Space Complexity: O(C).
    '''
    def employeeFreeTime(self, schedule: List[List[Interval]]) -> List[Interval]:
        # Flatten the schedule
        intervals = [interval for employee_times in schedule for interval in employee_times]

        # Sort by start of intervals
        intervals.sort(key=lambda x: x.start)

        # Prep variables
        result = []
        end = intervals[0].end

        # Check for free time between intervals, comparing start of next, with end of prev
        # Start from 2nd interval, because anything before that has an infinite interval
        for interval in intervals[1:]:
            if end < interval.start:
                result.append(Interval(end, interval.start))
            end = max(end, interval.end)

        return result