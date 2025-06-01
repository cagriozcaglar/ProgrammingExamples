'''
Leetcode 153: Find minimum in rotated sorted array

Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

[4,5,6,7,0,1,2] if it was rotated 4 times.
[0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

Example 1:
Input: nums = [3,4,5,1,2]
Output: 1
Explanation: The original array was [1,2,3,4,5] rotated 3 times.

Example 2:
Input: nums = [4,5,6,7,0,1,2]
Output: 0
Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4 times.

Example 3:
Input: nums = [11,13,15,17]
Output: 11
Explanation: The original array was [11,13,15,17] and it was rotated 4 times. 
'''
from typing import List

class Solution:
    # Compare current element nums[mid] to last element nums[right]
    def findMin(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while left < right:
            mid = (left + right) // 2
            if nums[mid] > nums[right]:
                # Min must be in A[mid + 1:right + 1].
                left = mid + 1
            else:  # nums[mid] < nums[right].
                # Min cannot be in A[mid + 1:right + 1] so it must be in A[left:mid + 1]
                right = mid
        # Loop ends when left == right.
        return nums[left]

    # Compare current element nums[mid] to first element nums[0]
    def findMin2(self, nums: List[int]) -> int:
        # If the list has just one element then return that element.
        if len(nums) == 1:
            return nums[0]

        left, right = 0, len(nums)-1

        # e.g. 1 < 2 < 3 < 4 < 5 < 7. Already sorted array.
        # Hence the smallest element is first element. A[0]
        if nums[right] > nums[0]:
            return nums[0]

        while left <= right:
            mid = (left + right) // 2

            # Did we find it?
            if nums[mid] > nums[mid+1]:
                return nums[mid+1]
            if nums[mid-1] > nums[mid]:
                return nums[mid]

            # Binary search
            if nums[mid] > nums[0]:
                left = mid+1
            elif nums[mid] < nums[0]:
                right = mid-1