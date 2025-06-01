'''
:Leetcode 53: Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example 1:
Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6

Example 2:
Input: nums = [1]
Output: 1
'''

from typing import List
class Solution: # Kadane's algorithm, DP
    def maxSubArray(self, nums: List[int]) -> int:
        """
        Solution using Kadane's Algorithm
        Time complexity: O(n)
        Space complexity: O(1)
        :param nums:
        :return:
        """
        current_subarray_sum = max_subarray_sum = nums[0]
        for num in nums[1:]:
            # If current subarray sum is negative, throw it away.
            # Otherwise, keep adding to it.
            current_subarray_sum = max(num, current_subarray_sum + num)
            max_subarray_sum = max(max_subarray_sum, current_subarray_sum)
    
        return max_subarray_sum