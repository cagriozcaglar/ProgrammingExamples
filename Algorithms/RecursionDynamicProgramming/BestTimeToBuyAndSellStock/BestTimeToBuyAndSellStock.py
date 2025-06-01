"""
Leetcode 121: Best Time to Buy and Sell Stock

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

"""
from typing import List
import sys
class Solution:
    '''
    The points of interest are the peaks and valleys in the given graph. We need to find the largest price following each valley, which difference could be the max profit.
    We can maintain two variables - minprice and maxprofit corresponding to the smallest valley and maximum profit (maximum difference between selling price and minprice) obtained so far respectively.
    - Time complexity: O(n). Only a single pass is needed.
    - Space complexity: O(1). Only two variables are used.
    '''
    def maxProfit(self, prices: List[int]) -> int:
        min_price = float("inf")
        max_profit = 0
        for i in range(len(prices)):
            if prices[i] < min_price:
                min_price = prices[i]
            elif prices[i] - min_price > max_profit:
                max_profit = prices[i] - min_price

        return max_profit

    def maxProfitKadane(self, prices: List[int]) -> int:
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