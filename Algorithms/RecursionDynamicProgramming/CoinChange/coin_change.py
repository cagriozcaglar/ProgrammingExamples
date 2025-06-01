'''
Leetcode 322: Coin Change

You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.

Example 1:
Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1

Example 2:
Input: coins = [2], amount = 3
Output: -1

Example 3:
Input: coins = [1], amount = 0
Output: 0
'''

'''
Approach 3: (Dynamic programming - Bottom up)
- Time complexity : O(S*n).
On each step the algorithm finds the next F(i) in n iterations, where 1≤i≤S. Therefore in total the iterations are S∗n.
- Space complexity : O(S).
We use extra space for the memoization table.
'''
from typing import List
import math

class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [math.inf] * (amount + 1)
        dp[0] = 0
        for am in range(1, amount+1):
            for coin in coins:
                if am - coin < 0:
                    continue
                dp[am] = min(dp[am], dp[am - coin] + 1)
        return dp[amount] if dp[amount] != math.inf else -1