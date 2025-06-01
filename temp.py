'''
Leetcode 57: Insert Interval

You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return intervals after the insertion.

Note that you don't need to modify intervals in-place. You can make a new array and return it.
'''
from typing import List

# https://leetcode.com/problems/insert-interval/discuss/844494/Python-O(n)-solution-explained
class Solution:
    # def insert(self, intervals, I):
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result, i = [], -1
        for i, (start, end) in enumerate(intervals):
            # current interval strictly before new interval: end < new.start => append
            if end < newInterval[0]:
                result.append([start, end])
            # current interval strictly after new interval: new.end < start => 
            elif newInterval[1] < start:
                i -= 1
                break
            # Else, intersection between current & new intervals: Update newInterval as:
            # newInterval = ( min(newInterval[0], start), max(newInterval[1], end))
            else:
                newInterval[0] = min(newInterval[0], start)
                newInterval[1] = max(newInterval[1], end)
                
        return result + [newInterval] + intervals[i+1:]