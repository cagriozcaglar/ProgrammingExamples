'''
Leetcode 713: Subarray Product Less Than K

Given an array of integers nums and an integer k, return the number of contiguous subarrays where the product of all the elements in the subarray is less than k.

Example 1:
Input: nums = [10,5,2,6], k = 100
Output: 8
Explanation: The 8 subarrays that have product less than 100 are:
[10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6]
Note that [10, 5, 2] is not included as the product of 100 is not strictly less than k.

Example 2:
Input: nums = [1,2,3], k = 0
Output: 0
'''
from typing import List

# Sliding Window solution
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        # Edge case: k is 0 or 1 (no subarrays possible)
        if k <= 1:
            return 0

        total_count = 0
        product = 1

        # Two pointers to maintain sliding window
        left = 0
        for right, num in enumerate(nums):
            # Expand window by including element at right pointer
            product *= num
            # Shrink window until product is smaller than k
            while product >= k:
                # Remove element at left pointer, proceed left pointer
                product //= nums[left]
                left += 1
            # Update total count by adding num of valid subarrays with current window size
            total_count += right - left + 1
        
        return total_count