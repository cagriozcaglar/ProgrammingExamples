"""
123. Best Time to Buy and Sell Stock III, In at most 2 Transactions
DP, O(n) time, O(n) space
"""
from typing import List
class Solution:
    def maxProfitDp(self, prices: List[int]) -> int:
        if len(prices) <= 1:
            return 0

        leftMin = prices[0]
        rightMax = prices[-1]

        length = len(prices)
        leftProfits = [0] * length
        rightProfits = [0] * (length+1)

        # Construct bidirectional DP array
        for left in range(1, length):
            # Left
            leftProfits[left] = max(leftProfits[left-1], prices[left]-leftMin)
            leftMin = min(leftMin, prices[left])

            # Right
            right = length - left - 1
            rightProfits[right] = max(rightProfits[right+1], rightMax - prices[right])
            rightMax = max(rightMax, prices[right])

        maxProfit = 0
        for i in range(0, length):
            maxProfit = max(maxProfit, leftProfits[i] + rightProfits[i+1])

        return maxProfit

    def maxProfit(self, prices: List[int]) -> int:
        t1Cost, t2Cost = float('inf'), float('inf')
        t1Profit, t2Profit = 0, 0

        for price in prices:
            # Maximum profit in only one transaction
            t1Cost = min(t1Cost, price)
            t1Profit = max(t1Profit, price - t1Cost)
            # Reinvest gained profit in second transaction
            t2Cost = min(t2Cost, price - t1Profit)
            t2Profit = max(t2Profit, price - t2Cost)

        return t2Profit