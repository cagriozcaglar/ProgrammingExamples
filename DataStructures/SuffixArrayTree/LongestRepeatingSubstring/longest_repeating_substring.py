'''
Leetcode 1062: Longest Repeating Substring

Given a string s, find out the length of the longest repeating substring(s). Return 0 if no repeating substring exists.

Example 1:
Input: s = "abcd"
Output: 0
Explanation: There is no repeating substring.

Example 2:
Input: s = "abbaba"
Output: 2
Explanation: The longest repeating substrings are "ab" and "ba", each of which occurs twice.

Example 3:
Input: s = "aabcaabdaab"
Output: 3
Explanation: The longest repeating substring is "aab", which occurs 3 times.
'''

class Solution:
    def longestRepeatingSubstring(self, s: str) -> int:
        # Create suffix array
        suffixes = [s[i:] for i in range(len(s))]
        # Sort suffixes array, in-place
        suffixes.sort()

        # Init max_length of repeating substring
        max_length = 0
        # *** Prefix of a suffix is a substring of a string ***
        # Compare adjacent suffixes, find the longest common prefix
        for i in range(1, len(s)):
            j = 0  # iterate over consecutive suffixes with pointer j
            # Compare chars until they differ or end of one suffix is reached
            while j < min(len(suffixes[i]), len(suffixes[i - 1])) and \
            suffixes[i][j] == suffixes[i - 1][j]:
                j += 1
            # Update max_length with the length of common prefix
            max_length = max(max_length, j)

        return max_length
