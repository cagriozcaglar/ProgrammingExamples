'''
Leetcode 14: Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string "".

Example 1:
Input: strs = ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: strs = ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.
'''
from typing import List

'''
Approach 2: Vertical scanning
Imagine a very short string is the common prefix at the end of the array. The above approach will still do S comparisons. One way to optimize this case is to do vertical scanning. We compare characters from top to bottom on the same column (same character index of the strings) before moving on to the next column.

- Time complexity : O(S) , where S is the sum of all characters in all strings.
  - In the worst case there will be n equal strings with length m and the algorithm performs S=m⋅n character comparisons.
  - Even though the worst case is still the same as Approach 1, in the best case there are at most n⋅minLen comparisons where minLen is the length of the shortest string in the array.
- Space complexity : O(1). We only used constant extra space.
'''
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""

        for i in range(len(strs[0])):
            c = strs[0][i]
            for j in range(1, len(strs)):
                if i == len(strs[j]) or strs[j][i] != c:
                    return strs[0][0:i]
        return strs[0]
