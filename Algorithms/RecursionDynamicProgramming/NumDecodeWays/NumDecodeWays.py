"""
91. Decode Ways
A message containing letters from A-Z can be encoded into numbers using the following mapping:
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the
mapping above (there may be multiple ways). For example, "11106" can be mapped into:
 1) "AAJF" with the grouping (1 1 10 6)
 2) "KJF" with the grouping (11 10 6)
Given a string s containing only digits, return the number of ways to decode it.
"""
# https://leetcode.com/problems/decode-ways/discuss/253018/Python%3A-Easy-to-understand-explanation-bottom-up-dynamic-programming
class Solution:
    def numDecodings(self, s: str) -> int:
        # dp[i] = number of ways to parse the string s[1: i + 1]
        dp = [0] * (len(s)+1)
        # Base cases
        dp[0] = 1
        dp[1] = 0 if s[0] == "0" else 1

        for i in range(2, len(s)+1):
            # One step jump
            if 1 <= int(s[i-1]) <= 9:
                dp[i] += dp[i-1]
            # Two step jump
            if 10 <= int(s[i-2:i]) <= 26:
                dp[i] += dp[i-2]

        return dp[-1]

    """
    # Numbers -> chars
      s = [s_1, s_2, ..., s_n]
      numDec(s) = numDec(s[1:]) + numDec(s[2:])
      
      numDec("") = 0
      numDec(s) = 1 ; |s| == 1
      numDec(s) = 2 ; |s| == 2 and s[0] != 0      
    """
    def numDecodingsDoesntWork(self, s: str) -> int:
        if not s:
            return 1
        if len(s) == 1 and s != "0":
            return 1
        count = 0
        if "1" <= s[0] <= "9":
            count += self.numDecodings(s[1:])
        if s[0] != "0":
            if 10 <= int(s[0:2]) <= 26:
                count += self.numDecodings(s[2:])
        return count