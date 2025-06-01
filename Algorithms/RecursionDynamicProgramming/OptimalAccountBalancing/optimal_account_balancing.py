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

'''
The intuition behind the solution is to first simplify the problem by calculating the net balance of each person after all the transactions. A net balance is the amount of money a person owes (negative balance) or is owed (positive balance). If a person has a net balance of zero, no further action is needed for them, so we ignore them in the settlement process.

To settle the debts, we aim to find the fewest number of transactions that would balance these net amounts. This becomes a combinatorial optimization problem where we explore different ways to match debtors (people with a negative balance) with creditors (people with a positive balance) in a minimal fashion. Essentially, it's a variation of the subset sum problem, which is computationally challenging (NP-Complete) due to the number of combinations to consider.

We use dynamic programming to manage the complexity. We represent each subset of people using a bitmask, where the i-th bit represents whether the i-th person is included in the subset. Our dynamic programming array f keeps track of the minimum number of transactions needed for each subset of people. The value f[i] contains the optimal (minimum) number of transactions needed to settle the debts among the subset of people represented by the bitmask i.

By iterating through all possible subsets, we try to construct solutions from combinations of smaller subsets that sum up to zero, meaning their debts can be settled among themselves. We use bit manipulation to iterate through subsets and to compute the number of bits set (number of people involved) in a state. The bit count of a state minus one gives us the correct number of transactions, assuming a direct transaction could settle the debts among people in that state.

The final answer is then found in f[-1], which represents the minimum number of transactions needed to settle all debts among all people, encoded in the bitmask with all bits set.
'''

from collections import defaultdict
import math
from typing import List

class Solution:
    def min_transfers(self, transactions) -> int:
        balance = defaultdict(int)
        # Calculate the net balance for each person
        for from_person, to_person, amount in transactions:
            balance[from_person] -= amount
            balance[to_person] += amount
      
        # Filter out people with a zero balance as they do not need any transfers
        debts = [amount for amount in balance.values() if amount]
        number_of_people = len(debts)
      
        # Initialize the dp array to store minimum transfers for each subset
        # set fewest_transfers[i] = inf for all i, except fewest_transfers[0] = 0
        fewest_transfers = [inf] * (1 << number_of_people)
        fewest_transfers[0] = 0
      
        # Evaluate each subset of people
        for i in range(1, 1 << number_of_people):
            sum_of_debts = 0
            # Calculate the sum of debts for the current subset
            for j, debt in enumerate(debts):
                if i >> j & 1:
                    sum_of_debts += debt
          
            # If the sum of debts is zero, it's possible to settle within the group
            if sum_of_debts == 0:
                # The number of transactions needed is the bit count of i (number of set bits) minus 1
                fewest_transfers[i] = bin(i).count('1') - 1
              
                # Try to split the subset in different ways and keep the minimum transfers
                subset = (i - 1) & i
                while subset > 0:
                    fewest_transfers[i] = min(fewest_transfers[i], fewest_transfers[subset] + fewest_transfers[i ^ subset])
                    subset = (subset - 1) & i
      
        # The answer is in fewest_transfers[-1] which corresponds to the situation where all people are considered
        return fewest_transfers[-1]

    # Dynamic Programing
    # Time complexity: O(n * 2^n)
    # Space complexity: O(2^n)
    def minTransfersDynamicProgramming(self, transactions: List[List[int]]) -> int:
        balance_map = defaultdict(int)
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
        balance_map = defaultdict(int)
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