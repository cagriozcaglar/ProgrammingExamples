"""
90. Subsets II (https://leetcode.com/problems/subsets-ii/)
Given an integer array nums that may contain duplicates, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
"""
from typing import List

class Solution:
    # Solution 1: Backtracking
    def subsetsWithDup_Backtracking(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        subsets = []
        currentSubset = []
        self.subsetsWithDupHelper(subsets, currentSubset, nums, 0)
        return subsets

    def subsetsWithDupHelper(self, subsets, currentSubset, nums, index):
        # Add the subset formed so far to the subsets list
        subsets.append(list(currentSubset))
        for i in range(index, len(nums)):
            # If current element is a duplicate, ignore. (***: Tough condition below)
            if i != index and nums[i] == nums[i-1]:
                continue
            currentSubset.append(nums[i])
            self.subsetsWithDupHelper(subsets, currentSubset, nums, i+1)
            currentSubset.pop()

    # Solution 2: Bitmasking
    def subsetsWithDup_Bitmask(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        # Sort generated subset. Will help identify duplicates
        nums.sort()
        maxNumberOfSubsets = 2**n

        # To store previously seen sets
        seen = set()
        subsets = []
        for subsetIndex in range(maxNumberOfSubsets):
            # Append subset corresponding to that bitmask
            currentSubset = []
            hashcode = ""
            for j in range(n):  # length from [0..n-1]
                # Generate bitmask
                mask = 2**j
                isSet = mask & subsetIndex
                if isSet != 0:
                    currentSubset.append(nums[j])
                    # Generate the hashcode by creating a comma-separated string
                    # of numbers in the currentSubset
                    hashcode += str(nums[j]) + ","
            if hashcode not in seen:
                subsets.append(currentSubset)
                seen.add(hashcode)

        return subsets