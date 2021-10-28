"""
Note: This is Leetcode 243: https://leetcode.com/problems/shortest-word-distance/

Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.

Example:
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Input: word1 = “coding”, word2 = “practice”
Output: 3

Input: word1 = "makes", word2 = "coding"
Output: 1

Note:
You may assume that word1 does not equal to word2, and word1 and word2 are both in the list.
"""

import sys
from typing import List


class ShortestWordDistance:
    @staticmethod
    def shortestDistance(words: List[str], word1: str, word2: str) -> int:
        """
        One-pass solution: Keep two indices index1 and index2 where we store the most recent locations of word1 and word2.
        Each time we find a new occurrence of one of the words, we do not need to search the entire array for the other
        word, since we already have the index of its most recent occurrence.

        - Time complexity: O(n), where n is the length of words list
        - Space complexity: O(1) (only two pointers and one minDistance variable are used.)
        :param words:
        :param word1:
        :param word2:
        :return:
        """
        INVALID_INDEX = -1
        index1 = INVALID_INDEX
        index2 = INVALID_INDEX
        # NOTE: sys.maxsize is how you get max integer value in Python3 onwards. Min value is min = -sys.maxsize - 1.
        minDistance: int = sys.maxsize

        for i in range(len(words)):
            if words[i] == word1:
                index1 = i
            if words[i] == word2:
                index2 = i
            # Before updating minDistance, check if both indices are updates & valid
            if index1 != INVALID_INDEX and index2 != INVALID_INDEX:
                minDistance = min(minDistance, abs(index1-index2))
        return minDistance


if __name__ == "__main__":
    """
    Example:
    Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

    Input: word1 = “coding”, word2 = “practice”
    Output: 3
    
    Input: word1 = "makes", word2 = "coding"
    Output: 1
    """
    words = ["practice", "makes", "perfect", "coding", "makes"]
    # Example 1
    word1 = "coding"
    word2 = "practice"
    print(f"Shortest distance between \"{word1}\" and \"{word2}\" in word list {words}: "
          f"{ShortestWordDistance.shortestDistance(words, word1, word2)}")
    # Shortest distance between "coding" and "practice" in word list ['practice', 'makes', 'perfect', 'coding', 'makes']: 3

    # Example 2
    word1 = "makes"
    word2 = "coding"
    print(f"Shortest distance between \"{word1}\" and \"{word2}\" in word list {words}: "
          f"{ShortestWordDistance.shortestDistance(words, word1, word2)}")
    # Shortest distance between "makes" and "coding" in word list ['practice', 'makes', 'perfect', 'coding', 'makes']: 1
