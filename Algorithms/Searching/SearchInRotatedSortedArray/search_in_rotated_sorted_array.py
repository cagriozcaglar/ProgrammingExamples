'''
Leetcode 33: Search in Rotated Sorted Array

There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot
index k (1 <= k < nums.length) such that the resulting array is [nums[k],
nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed).
For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of
target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1

Example 3:
Input: nums = [1], target = 0
Output: -1
'''
from typing import List

class Solution:
    # Approach 3: One Binary Search
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        while left <= right:
            mid = (left + right) // 2

            # Case 1: Found target
            if nums[mid] == target:
                return mid

            # Case 2: Subarray on mid's left is sorted
            elif nums[mid] >= nums[left]:
                if target >= nums[left] and target < nums[mid]:  # Search left
                    right = mid - 1
                else:  # Search right
                    left = mid + 1

            # Case 3: Subarray on mid's right is sorted
            else:
                if target <= nums[right] and target > nums[mid]:  # Search right
                    left = mid + 1
                else:  # Search left
                    right = mid - 1

        return -1

    # Approach 2: Find Pivot Index + Binary Search with Shift
    def search2(self, nums: List[int], target: int) -> int:
        n = len(nums)
        left, right = 0, n - 1
        # Find the index of the pivot
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] > nums[-1]:  # start is between [mid+1, right]
                left = mid + 1
            else:
                right = mid - 1

        # Shift elements in circular manner, with pivot at index 0. Perform BS.
        def shiftedBinarySearch(pivot_index, target):
            shift = n - pivot_index
            left, right = (pivot_index + shift) % n, (pivot_index-1+shift) % n

            while left <= right:
                mid = (left + right) // 2
                if nums[(mid - shift) % n] == target:
                    return (mid - shift) % n
                elif nums[(mid - shift) % n] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return -1

        return shiftedBinarySearch(left,target)