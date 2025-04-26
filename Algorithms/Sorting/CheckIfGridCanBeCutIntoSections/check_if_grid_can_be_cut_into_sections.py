'''
Leetcode 3394. Check if Grid Can Be Cut Into Sections

You are given an integer n representing the dimensions of an n x n grid,
with the origin at the bottom-left corner of the grid.
You are also given a 2D array of coordinates rectangles, where rectangles[i]
is in the form [startx, starty, endx, endy], representing a rectangle on the grid.
Each rectangle is defined as follows:

 * (startx, starty): The bottom-left corner of the rectangle.
 * (endx, endy): The top-right corner of the rectangle.

Note that the rectangles do not overlap. Your task is to determine if it is possible to make either two horizontal or two vertical cuts on the grid such that:

Each of the three resulting sections formed by the cuts contains at least one rectangle.
Every rectangle belongs to exactly one section.
Return true if such cuts can be made; otherwise, return false.

'''
# Time complexity: O(n*logn)
# Space complexity: O(S) (S: Sorting algorithm S's space complexity)

from typing import List

class Solution:
    def checkValidCuts(self, n: int, rectangles: List[List[int]]) -> bool:
        # Generate x-axis and y-axis intervals.
        # Sort both by starting point.
        # Check if intervals on any of the axis can be split into 3 non-overlapping parts.
        '''
        [s_x_1, e_x_1], ..., [s_x_1, e_x_n]
        Merge consecutive intervals:
        1) If two consecutive intervals overlap, merge and move to next.
        2) If intervals do not overlap, that is a place to split, start a new interval, increment count.

        After processing all intervals, if num_sections exceeds 2, stop and return true.
        Otherwise, by default, return false
        '''
        # X-axis intervals, sorted
        x_axis_intervals: List[List[int]] = [[dims[0], dims[2]] for dims in rectangles]
        x_axis_intervals.sort(key=lambda x: x[0])

        # Y-axis intervals, sorted
        y_axis_intervals: List[List[int]] = [[dims[1], dims[3]] for dims in rectangles]
        y_axis_intervals.sort(key=lambda x: x[0])

        def can_cut_into_sections(intervals, n=3):
            merged_intervals = []
            if len(intervals) < 2:
                return False
            merged_intervals = [intervals[0]]
            for i in range(1, len(intervals)):
                [s1, e1] = merged_intervals[-1]
                [s2, e2] = intervals[i]
                if s2 < e1:  # merge
                    # IMPORTANT: When merging, you returned [s1, e2)],
                    # but it should be [s1, max(e1, e2)]
                    # Because we don't have any ordering between e1 and e2!
                    merged_intervals[-1] = [s1, max(e1, e2)]
                else:  # start new section
                    merged_intervals.append(intervals[i])
                if len(merged_intervals) >= n:
                    return True

            return False

        return can_cut_into_sections(x_axis_intervals) or can_cut_into_sections(y_axis_intervals)