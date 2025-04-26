'''
Leetcode 46: permutations
Given an array nums of distinct integers, return all the possible permutations.
You can return the answer in any order.
'''
from typing import List

class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def directed_perms(i):
            # Base case: permutation found, hitting length end
            if i == len(nums) - 1:
                result.append(nums.copy())
                return

            # Make move, backtrack, unmake move
            # For all possible indices j in range [i, len(A)-1]
            for j in range(i, len(nums)):
                # 1. Make move: swap A[i] <-> A[j]
                nums[i], nums[j] = nums[j], nums[i]
                # 2. Backtrack: proceed to next character
                directed_perms(i + 1)
                # 3. Unmake move: reverse swap A[j] <-> A[i]
                nums[i], nums[j] = nums[j], nums[i]

        result = []
        directed_perms(0)
        return result


    def permute2(self, nums: List[int]) -> List[List[int]]:
        def backtrack(curr):
            if len(curr) == len(nums):
                answer.append(curr[:])
                return

            for num in nums:
                if num not in curr:
                    # Make move
                    curr.append(num)
                    # Backtrack
                    backtrack(curr)
                    # Unmake move
                    curr.pop()

        answer = []
        backtrack([])
        return answer
        