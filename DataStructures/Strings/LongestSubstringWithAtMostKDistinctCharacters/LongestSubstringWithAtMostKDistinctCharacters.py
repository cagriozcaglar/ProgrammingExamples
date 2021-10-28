"""
Note: This is Leetcode question 340, Longest Substring with At Most K Distinct Characters: https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/

Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.

Example 1:
Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.

Example 2:
Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.


Constraints:
 - 1 <= s.length <= 5 * 104
 - 0 <= k <= 50

"""


class LongestSubstringWithAtMostKDistinctCharacters:
    @staticmethod
    def lengthOfLongestSubstringKDistinct(s: str, k: int) -> int:
        """
        Sliding window with two pointers, start and end.
        We also use a dictionary of characters to counts for the current substring s[start..end]

        Pseudocode:
          longestLength = 0   // max length variable
          start = 0, end = 0  // two pointers
          charCountMap = {}   // dictionary of characters to counts for the current substring s[start..end]
          while end != len(s):
            charCountMap[s[end]] += 1   // Add / increment count of s[end] in charCountMap
            // Check if there are more than k distinct characters in new s[start..end]
            while len(charCountMap) > k:
              charCountMap[s[start]] -= 1  // Remove / decrement count of s[start] in charCountMap
              start++ // Increment start pointer
            longestLength = max(longestLength, end-start+1)
            end++ // Increment end pointer
          return longestLength

        Time Complexity: O(n). Start and end pointers are increased with atomic operations of O(1), until they hit max length n.
        Space complexity: O(k). Additional hashmap can become of size k at max. Start and end variables are O(1).

        :param s: string
        :param k: max number of distinct characters
        :return:
        """
        # Longest length set to 0
        longestLength = 0
        # Map of characters to counts for s[start..end]
        characterCountMap = {}
        # Two pointers: Sliding window approach
        start = 0
        end = 0

        while end != len(s):
            # NOTE: In case key does not exist in hashmap, on RHS, we use .get(key, 0) where 0 is default, and on LHS
            # we still access the map using index characterCountMap[key]. (key here is s[end]).
            characterCountMap[s[end]] = characterCountMap.get(s[end], 0) + 1
            # If there are more than k distinct characters, keep increasing start
            while len(characterCountMap) > k:
                characterCountMap[s[start]] = characterCountMap.get(s[start], 0) - 1
                # If s[start] maps to count of 0, remove it from the map
                if characterCountMap[s[start]] <= 0:
                    characterCountMap.pop(s[start])
                start = start + 1
            longestLength = max(longestLength, end-start+1)
            end = end + 1

        return longestLength


if __name__ == "__main__":
    """
    Example 1:
    Input: s = "eceba", k = 2
    Output: 3
    Explanation: The substring is "ece" with length 3.
    """
    s = "eceba"
    k = 2
    print(f"Length of longest substring of {s} with at most {k} distinct characters is "
          f"{LongestSubstringWithAtMostKDistinctCharacters.lengthOfLongestSubstringKDistinct(s,k)}.")

    """
    Example 2:
    Input: s = "aa", k = 1
    Output: 2
    Explanation: The substring is "aa" with length 2.
    """
    s = "aa"
    k = 1
    print(f"Length of longest substring of {s} with at most {k} distinct characters is "
          f"{LongestSubstringWithAtMostKDistinctCharacters.lengthOfLongestSubstringKDistinct(s,k)}.")
