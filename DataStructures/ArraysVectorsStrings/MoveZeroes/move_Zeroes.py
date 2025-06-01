'''
Leetcode 283: Move Zeroes

Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Note that you must do this in-place without making a copy of the array.
'''
from typing import List

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        This function takes a list of numbers and moves all the zeros to the end,
        maintaining the relative order of the other elements.
        """
        # last_non_zero_found_at keeps track of the last non-zero index found in the array 
        last_non_zero_found_at = 0

        # Iterate over the array
        for current, value in enumerate(nums):
            # When a non-zero element is found
            if value != 0:
                # Swap the current non-zero element with the element at last_non_zero_found_at index
                nums[last_non_zero_found_at], nums[current] = nums[current], nums[last_non_zero_found_at]
                # Move the last_non_zero_found_at index forward
                last_non_zero_found_at += 1

        # Note: This function does not return anything as it is supposed to modify the nums list in-place.