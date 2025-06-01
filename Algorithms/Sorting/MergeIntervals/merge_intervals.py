'''
Leetcode 56: Merge Intervals

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
'''
from typing import List

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        merged_intervals = []

        # 1. Sort intervals by increasing order of starting point
        intervals.sort(key=lambda interval: interval[0])
        if intervals:
            # Done after sorting above: You need the first interval with min start
            merged_intervals = [intervals[0]]
        # 2. Iterate over intervals in increasing order of starting point.
        for i in range(1, len(intervals)):
            (s1, e1), (s2, e2) = merged_intervals[-1], intervals[i]
            # If s2 < e1 => merge
            if s2 <= e1:
                # Merge process: min(start points), max(end points)
                merged_intervals[-1] = [min(s1, s2), max(e1, e2)]
            else:  # s2 > e2 => non-overlapping
                merged_intervals.append(intervals[i])

        return merged_intervals