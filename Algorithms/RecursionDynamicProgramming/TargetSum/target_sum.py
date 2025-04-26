'''

Leetcode 494: Target Sum

You are given an integer array nums and an integer target.

You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
Return the number of different expressions that you can build, which evaluates to target.

Example 1:
Input: nums = [1,1,1,1,1], target = 3
Output: 5
Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
-1 + 1 + 1 + 1 + 1 = 3
+1 - 1 + 1 + 1 + 1 = 3
+1 + 1 - 1 + 1 + 1 = 3
+1 + 1 + 1 - 1 + 1 = 3
+1 + 1 + 1 + 1 - 1 = 3

Example 2:
Input: nums = [1], target = 1
Output: 1
'''
from typing import List
import math

class Solution4DynamicProgrammingSpaceOptimized:
    # Solution 4: Dymamic Programming, Space Optimized
    # Each entry in the table is calculated based on the entries from the previous row.
    # No need to keep 2D memo table, only 1 row is sufficient.
    # Time: O(n * total_sum)
    # Space: O(2 * total_sum)
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        # dp is of size (2*total_sum + 1) (only last row is kept)
        dp = [0] * (2 * total_sum + 1)

        # Init first row of DP table
        dp[nums[0] + total_sum] = 1    # Adding nums[0]
        dp[-nums[0] + total_sum] += 1  # Subtracting nums[0]

        # Fill the DP table
        for index in range(1, len(nums)):
            next_dp = [0] * (2 * total_sum + 1)
            for sum_val in range(-total_sum, total_sum + 1):
                if dp[sum_val + total_sum] > 0:
                    next_dp[sum_val + nums[index] + total_sum] += dp[sum_val + total_sum]
                    next_dp[sum_val - nums[index] + total_sum] += dp[sum_val + total_sum]
            dp = next_dp

        return 0 if abs(target) > total_sum else dp[target + total_sum]


class Solution3DynamicProgramming:
    # Solution 3: Dymamic Programming
    # Bottom-up dymamic programming
    # Time: O(n * total_sum)
    # Space: O(n * total_sum)
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        total_sum = sum(nums)
        # dp is of size len(nums) * (2*total_sum + 1)
        dp = [ [0] * (2 * total_sum + 1) for _ in range(len(nums)) ]

        # Init first row of DP table
        dp[0][nums[0] + total_sum] = 1
        dp[0][-nums[0] + total_sum] += 1

        # Fill the DP table
        for index in range(1, len(nums)):
            for sum_val in range(-total_sum, total_sum + 1):
                if dp[index-1][sum_val + total_sum] > 0:
                    dp[index][sum_val + nums[index] + total_sum] += dp[index-1][sum_val + total_sum]
                    dp[index][sum_val - nums[index] + total_sum] += dp[index-1][sum_val + total_sum]

        return 0 if abs(target) > total_sum else dp[len(nums)-1][target + total_sum]



class Solution2RecursionMemoization:
    # Solution 2: Recursion with Memoization
    # Subproblem: Reaching a sum with first k numbers
    # Memoization: memo[index][currentSum]
    # Time: O(n * total_sum)
    # Space: O(n * total_sum)
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        self.total_sum = sum(nums)
        # sum range is [-total_sum, + total_sum], which has (2*total_sum + 1) values
        memo = [ [-math.inf] * (2 * self.total_sum + 1) for _ in range(len(nums)) ]
        return self.calculate_ways(nums, 0, 0, target, memo)

    def calculate_ways(self, nums: List[int], current_index: int, current_sum: int, target: int, memo: List[List[int]]) -> int:
        if current_index == len(nums):
            return 1 if current_sum == target else 0
        else:
            # Check if result is in memo
            if memo[current_index][current_sum + self.total_sum] != -math.inf:
                return memo[current_index][current_sum + self.total_sum]
            # Add current number
            add = self.calculate_ways(nums, current_index + 1, current_sum + nums[current_index], target, memo)
            # Subtract current number
            sub = self.calculate_ways(nums, current_index + 1, current_sum - nums[current_index], target, memo)
            # Store result in memo
            memo[current_index][current_sum + self.total_sum] = add + sub

            return memo[current_index][current_sum + self.total_sum]

class Solution1BruteForce:
    # Solution 1: Brute force
    # Time: O(2^n)
    # Space: O(n)
    def __init__(self):
        self.total_ways = 0

    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        self.calculate_ways(nums, 0, 0, target)
        return self.total_ways

    def calculate_ways(self, nums: List[int], current_index: int, current_sum: int, target: int) -> int:
        # nonlocal total_ways
        if current_index == len(nums):
            if current_sum == target:
                self.total_ways += 1
        else:
            # Use positive sign
            self.calculate_ways(nums, current_index + 1, current_sum + nums[current_index], target)
            # Use negative sign
            self.calculate_ways(nums, current_index + 1, current_sum - nums[current_index], target)
