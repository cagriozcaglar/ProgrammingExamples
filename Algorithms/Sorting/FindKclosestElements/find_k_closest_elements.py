'''
Leetcode 658: Find K Closest Elements

Given a sorted integer array arr, two integers k and x, return the k closest integers
to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:

1) |a - x| < |b - x|, or
2) |a - x| == |b - x| and a < b

Example 1:
Input: arr = [1,2,3,4,5], k = 4, x = 3
Output: [1,2,3,4]

Example 2:
Input: arr = [1,2,3,4,5], k = 4, x = -1
Output: [1,2,3,4]
'''
from typing import List

class Solution:
    # Method 1: Sort with custom comparator
    # Time complexity: O(n*log(n) + k*log(k))
    # Space complexity: O(n)
    def findClosestElements1(self, arr: List[int], k: int, x: int) -> List[int]:
        # Sort using custom comparator
        sorted_array = sorted(arr, key=lambda num: abs(x - num))

        # Take first k elements, sort (by numeric value, not by distance to x)
        return sorted(sorted_array[:k])

    # Method 2: Binary search to find the left bound
    # Time complexity: O(log(N-k) + k)
    # Space complexity: O(1)
    def findClosestElements(self, arr: List[int], k: int, x: int) -> List[int]:
        # Init binary search bounds
        left = 0
        right = len(arr) - k

        # Binary search against the criteria 
        while left < right:
            mid = (left + right) // 2
            # Closer to left
            if x - arr[mid] > arr[mid + k] - x:
                left = mid + 1
            # Closer to right
            else:
                right = mid

        return arr[left: left + k]
