'''
Leetcode 198: House Robber

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

'''
from typing import List

'''
Space-optimized Dynamic Programming
- Time: O(n)
- Space: O(1)
'''
class Solution:
    def rob(self, nums: List[int]) -> int:
        # Empty case
        if not nums:
            return 0

        # Init running two numbers
        n = len(nums)
        rob_next_plus_one, rob_next = 0, nums[n - 1]

        # DP table calculations
        for i in range(n - 2, -1, -1):
            # Same as recursive solution
            current = max(rob_next, rob_next_plus_one + nums[i])

            # Update variables
            rob_next_plus_one = rob_next
            rob_next = current

        return rob_next

'''
Dynamic Programming
- Time: O(n)
- Space: O(n)
'''
class Solution3:
    def rob(self, nums: List[int]) -> int:
        # Empty case
        if not nums:
            return 0

        max_robbed_amount = [None for _ in range(len(nums) + 1)]
        n = len(nums)

        # Base case init
        max_robbed_amount[n], max_robbed_amount[n - 1] = 0, nums[n - 1]

        # DP table calculations
        for i in range(n - 2, -1, -1):
            # Same as recursive solution
            max_robbed_amount[i] = max(
                max_robbed_amount[i + 1],
                max_robbed_amount[i + 2] + nums[i]
            )

        return max_robbed_amount[0]

'''
Recursion with Memoization
- Time: O(n)
- Space: O(n)
'''
class Solution2:
    def __init__(self):
        self.memo = {}

    def rob(self, nums: List[int]) -> int:
        self.memo = {}
        return self.rob_from(0, nums)

    def rob_from(self, i, nums):
        # No more houses left
        if i >= len(nums):
            return 0

        # Return cached value
        if i in self.memo:
            return self.memo[i]

        # Recursive relation evaluation to get the optimal answer
        answer = max(
            self.rob_from(i + 1, nums),
            self.rob_from(i + 2, nums) + nums[i]
        )

        # Cache for future use
        self.memo[i] = answer
        return answer