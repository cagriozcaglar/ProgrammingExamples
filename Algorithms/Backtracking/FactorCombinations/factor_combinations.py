'''
Leetcode 254: Factor Combinations
Numbers can be regarded as the product of their factors.

For example, 8 = 2 x 2 x 2 = 2 x 4.
Given an integer n, return all possible combinations of its factors. You may return the answer in any order.

Note that the factors should be in the range [2, n - 1].

Example 1:

Input: n = 1
Output: []
Example 2:

Input: n = 12
Output: [[2,6],[3,4],[2,2,3]]
Example 3:

Input: n = 37
Output: []
'''

from typing import List
import math

# From AlgoMonster: https://algo.monster/liteproblems/254
# Time Complexity: O(n^1.5)
# Space Complexity: O(log(n))
class Solution:
    def getFactors(self, n: int) -> List[List[int]]:
        def get_directed_factors(num_remaining: int, current_factor: int) -> None:
            # If temp_factors has elements, then add a combination to the answer
            if temp_factors:
                all_factors.append(temp_factors + [num_remaining])
            # Init a factor to start from
            factor = current_factor
            # Check for factors up to sqrt of n
            while factor <= math.sqrt(num_remaining):
                if num_remaining % factor == 0:
                    # Make move: Append factor to temp list for possible result
                    temp_factors.append(factor)
                    # Backtracking: Recurse with reduced number (integer division)
                    get_directed_factors(num_remaining // factor, factor)
                    # Unmake move: Pop last factor to backtrack
                    temp_factors.pop()
                # Increment the factor
                factor += 1

        # Temporary list of factors for a combination
        temp_factors = []
        # Final list of lists to be returned
        all_factors: List[List[int]] = []
        # Start DFS / backtracking with whole number n, and smallest factor 2
        get_directed_factors(n, 2)
        return all_factors