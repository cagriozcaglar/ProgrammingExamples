'''
Leetcode 384: Shuffle an array

Given an integer array nums, design an algorithm to randomly shuffle the array. All permutations of the array should be equally likely as a result of the shuffling.

Implement the Solution class:

Solution(int[] nums) Initializes the object with the integer array nums.
int[] reset() Resets the array to its original configuration and returns it.
int[] shuffle() Returns a random shuffling of the array.
'''
# Time complexity: O(n)
# Space complexity: O(n)

from typing import List
import random

class Solution:

    def __init__(self, nums: List[int]):
        self.array = nums
        self.original_list = list(nums)

    def reset(self) -> List[int]:
        self.array = self.original_list
        self.original_list = list(self.original_list)
        return self.array

    def shuffle(self) -> List[int]:
        length = len(self.array)
        for i in range(length):
            index = random.choice(list(range(i, length)))
            self.array[i], self.array[index] = self.array[index], self.array[i]
        return self.array