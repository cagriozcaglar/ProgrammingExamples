"""
DP.
 - Time complexity: O(n*k), where:
    - n is the length of the prices sequence,
    - k: Max number of transactions

1. Keep holding the stock:
   dp[i][j][1] = dp[i−1][j][1]
2. Keep not holding the stock:
   dp[i][j][0] = dp[i−1][j][0]
3. Buying, when j>0:
   dp[i][j][1] = dp[i−1][j−1][0]−prices[i]
4. Selling:
   dp[i][j][0]=dp[i−1][j][1]+prices[i]

We can combine they together to find the maximum profit:
 1. dp[i][j][1]=max(dp[i−1][j][1],dp[i−1][j−1][0]−prices[i])
 2. dp[i][j][0]=max(dp[i−1][j][0],dp[i−1][j][1]+prices[i])
"""
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        # Edge cases
        if not prices or k==0:
            return 0

        # If 2*k > n, you are free to use any number of transactions
        if 2*k > n:
            result = 0
            for i, j in zip(prices[1:], prices[:-1]):
                result += max(0, i-j)
            return result

        # dp[i][j][l] = balance
        # i: day index (index of prices array)
        # j: # remaining transactions
        # l: # stocks on hand (0 or 1)

        # Init: 3-d array init
        dp = [ [ [-math.inf] * 2 for _ in range(k+1)] for _ in range(n)]

        # Base case:
        dp[0][0][0] = 0  # 0-th day, no transactions, no stock
        dp[0][1][1] = -prices[0]  # Buy first stock on day 0

        # Fill array
        for i in range(1,n):
            for j in range(k+1):
                # Transition equation:
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j][1]+prices[i])
                if j > 0:
                    # You can't hold stock without any transaction
                    dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])

        result = max(dp[n-1][j][0] for j in range(k+1))
        return result