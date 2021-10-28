"""
1269. Number of Ways to Stay in the Same Place After Some Steps
Dynamic Programming, Bottom-Up Fashion
"""
"""
dp(i,j): i-th step, j-th position of array

dp(0,0) = 1

dp(i,j) to:
  1) dp(i-1, j)  # stay
  2) dp(i-1, j+1) # move right
  3) dp(i-1, j-1) # move left

If arrLen is too big, doesn't make sense, set arrLen to min(steps+1, arrLen)
# Module 10^9 +7 at all times
"""
# Solution: https://code.dennyzhang.com/number-of-ways-to-stay-in-the-same-place-after-some-steps

MODULO = 10**9 + 7
class Solution:
    def numWays(self, steps: int, arrLen: int) -> int:
        arrLen = min(steps+1, arrLen)
        # Init
        dp = [ [0 for _ in range(arrLen)] for _ in range(steps+1) ]
        # Base condition
        dp[0][0] = 1
        # Bottom-up DP
        for i in range(1, len(dp)):
            for j in range(len(dp[0])):
                # 1. Stay
                dp[i][j] = dp[i-1][j]
                # 2. Left
                if j-1 >= 0:
                    dp[i][j] += dp[i-1][j-1]
                # 3. Right
                if j+1 < len(dp[0]):
                    dp[i][j] += dp[i-1][j+1]
                dp[i][j] %= MODULO
        # Return dp[-1][0]: Last step (-1), first index (0)
        return dp[-1][0]