'''
Leetcode 256. Paint Houses

There is a row of n houses, where each house can be painted one of three colors:
red, blue, or green. The cost of painting each house with a certain color is different.
You have to paint all the houses such that no two adjacent houses have the same color.

The cost of painting each house with a certain color is represented by an n x 3 cost matrix costs.

Return the minimum cost to paint all houses.

Input: costs = [
    [17,2,17],
    [16,16,5],
    [14,3,19]
]
Output: 10
Explanation: Paint house 0 into blue, paint house 1 into green, paint house 2 into blue.
Minimum cost: 2 + 5 + 3 = 10.
'''
from typing import List
from enum import Enum

class Color(Enum):
    RED = 0
    BLUE = 1
    GREEN = 2


# IMPORTANT: This is more optimal to solve with Dynamic Programming, rather than Memoization.
class Solution:
    # Time complexity: O(n)
    # Space complexity: O(1)
    def minCost_dynamicprogramming(self, costs: List[List[int]]) -> int:
        if not costs:
            return 0
        
        for n in reversed(range(len(costs)-1)):
            # n-th house is red
            costs[n][0] += min(costs[n+1][1], costs[n+1][2])
            # n-th house is green
            costs[n][1] += min(costs[n+1][0], costs[n+1][2])
            # n-th house is blue
            costs[n][2] += min(costs[n+1][0], costs[n+1][1])
        
        # Return the minimum in the first row
        return min(costs[0])

    # Time complexity: O(n)
    # Space complexity: O(n)
    def minCost_memoization(self, costs: List[List[int]]) -> int:
        self.cache = {}  # takes in (n, color) pair
        self.colors = {0, 1, 2}

        def paint_cost(n: int, color: int) -> int:
            if (n, color) in self.cache:
                return self.cache[(n, color)]
            
            total_cost = costs[n][color]

            if n < len(costs) - 1:
                non_colors = list(self.colors - set([color]))
                total_cost += min(*[paint_cost(n+1, non_color) for non_color in non_colors])
            
            self.cache[(n, color)] = total_cost
            return total_cost

        # Starts here
        if not costs:
            return 0
        
        return min(*[paint_cost(0, color) for color in [0,1,2]])