'''
Leetcode 5: Longest Palindromic Substring

Given a string s, return the longest palindromic substring in s.
'''

class Solution:
    # Approach 3 of Leetcode solution: Expand from centers
    # There are O(n^2) bounds, but only O(n) centers!
    # Time: O(n^2)
    # Space: O(1).  *** Improvement over Approach 2 with DP.
    def longestPalindrome(self, s: str) -> str:
        def expand(i, j):
            left = i
            right = j

            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1

            return right - left - 1

        ans = [0, 0]
        
        for i in range(len(s)):
            # Handle odd-length palindrome candidates
            odd_length = expand(i, i)
            if odd_length > ans[1] - ans[0] + 1:
                dist = odd_length // 2
                ans = [i-dist, i+dist]

            # Handle event-length palindrome candidates
            even_length = expand(i, i+1)
            if even_length > ans[1] - ans[0] + 1:
                dist = (even_length // 2) - 1
                ans = [i-dist, i+1+dist]

        i, j = ans
        return s[i: j+1]
            

    # Approach 2 of Leetcode solution: Dynamic Programming
    # Time: O(n^2)
    # Space: O(n^2) (dp table takes O(n^2) space).
    def longestPalindromeApproach2(self, s: str) -> str:
        n = len(s)
        dp = [[False]*n for _ in range(n)]
        ans = [0, 0]  # Strings of length 1 are palindromes

        # Init dp - part 1: substrings of length 1 are palindromes: dp[i][i] = True
        for i in range(n):
            dp[i][i] = True
        
        # Init dp - part 2: substrings of length 2, check equality
        for i in range(n-1):
            if s[i] == s[i+1]:
                dp[i][i+1] = True
                ans = [i, i+1]
        
        # Iterate over increasing diffs / sizes and update values.
        # diff from 2 to n-1
        for diff in range(2, n):
            for i in range(n - diff):
                j = i + diff
                if s[i] == s[j] and dp[i+1][j-1]:
                    dp[i][j] = True
                    ans = [i, j]
        
        i, j = ans
        return s[i: j+1] # j+1, because the end index is exclusive.