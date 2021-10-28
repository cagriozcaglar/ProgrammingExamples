"""
Best time to buy and sell stock
Kadane's algorithm, single pass, O(n)
"""
from typing import List
import sys
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxDiffSoFar = 0
        currentDiff = 0

        for index, price in enumerate(prices[1:]):  # CAREFUL: start array from index 1
            currentDiff += price - prices[index]  # CAREFUL: not index-1
            if currentDiff < 0:
                currentDiff = 0
            if maxDiffSoFar < currentDiff:
                maxDiffSoFar = currentDiff
        return maxDiffSoFar

    def maxProfitOld(self, prices: List[int]) -> int:
        # diffs = [secondPrice - firstPrice for firstPrice, secondPrice in zip(prices[:-1], prices[1:]) ]
        minPrice = sys.maxsize
        maxProfit = 0

        for price in prices:
            if price < minPrice:
                minPrice = price
            elif price - minPrice > maxProfit:
                maxProfit = price - minPrice

        return maxProfit