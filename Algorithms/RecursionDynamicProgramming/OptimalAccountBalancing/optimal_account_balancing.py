'''
Leetcode 465: Optimal Account Balancing

You are given an array of transactions transactions where transactions[i] = [fromi, toi, amounti] indicates that the person with ID = fromi gave amounti $ to the person with ID = toi.

Return the minimum number of transactions required to settle the debt.

Example 1:
Input: transactions = [[0,1,10],[2,0,5]]
Output: 2

Example 2:
Input: transactions = [[0,1,10],[1,0,1],[1,2,5],[2,0,5]]
Output: 1
'''

import collections
import math
from typing import List

class Solution:
    # Dynamic Programing
    # Time complexity: O(n * 2^n)
    # Space complexity: O(2^n)
    def minTransfersDynamicProgramming(self, transactions: List[List[int]]) -> int:
        balance_map = collections.defaultdict(int)
        for a, b, amount in transactions:
            balance_map[a] += amount
            balance_map[b] -= amount

        balance_list = [amount for amount in balance_map.values() if amount]
        n = len(balance_list)

        memo = [-1] * (1 << n)
        memo[0] = 0

        def dfs(total_mask):
            if memo[total_mask] != -1:
                return memo[total_mask]
            balance_sum, answer = 0, 0

            # Remove one person at a time in total_mask
            for i in range(n):
                cur_bit = 1 << i
                if total_mask & cur_bit:
                    balance_sum += balance_list[i]
                    answer = max(answer, dfs(total_mask ^ cur_bit))

            # If the total balance of total_mask is 0, increment answer by 1.
            memo[total_mask] = answer + (balance_sum == 0)
            return memo[total_mask]

        return n - dfs((1 << n) - 1)

    # Backtracking
    # Time complexity: O((n−1)!)
    # Space complexity: O(n)
    def minTransfersBacktracking(self, transactions: List[List[int]]) -> int:
        balance_map = collections.defaultdict(int)
        for a, b, amount in transactions:
            balance_map[a] += amount
            balance_map[b] -= amount

        balance_list = [amount for amount in balance_map.values() if amount]
        n = len(balance_list)

        def dfs(cur: int) -> int:
            while cur < n and not balance_list[cur]:
                cur += 1
            if cur == n:
                return 0
            cost = math.inf
            for nxt in range(cur + 1, n):
                # If next is valid recipient:
                # 1. Add cur's balance to nxt
                # 2. Recursively call dfs(cur+1)
                # 3. Remove cur's balance from nxt
                if balance_list[nxt] * balance_list[cur] < 0:
                    balance_list[nxt] += balance_list[cur]
                    cost = min(cost, 1 + dfs(cur + 1))
                    balance_list[nxt] -= balance_list[cur]
            return cost

        return dfs(0)