'''
Leetcode 34: Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]

Example 3:
Input: nums = [], target = 0
Output: [-1,-1]
'''
from typing import List

class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        lower_bound = self.findBound(nums, target, isFirst=True)
        if lower_bound == -1:
            return [-1, -1]

        upper_bound = self.findBound(nums, target, isFirst=False)

        return [lower_bound, upper_bound]

    def findBound(self, nums: List[int], target: int, isFirst: bool) -> int:
        left, right = 0, len(nums) - 1

        while left <= right:  # Double check
            mid = (left + right) // 2
            if nums[mid] == target:
                if isFirst:  # Finding first index
                    # Found lower bound
                    if mid == left or nums[mid-1] < target:
                        return mid
                    # Search left side for the bound
                    right = mid - 1
                else:  # Finding last index
                    # Found upper bound
                    if mid == right or nums[mid+1] > target:
                        return mid
                    # Search right side for the bound
                    left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:  # nums[mid] < target
                left = mid + 1

        return -1
