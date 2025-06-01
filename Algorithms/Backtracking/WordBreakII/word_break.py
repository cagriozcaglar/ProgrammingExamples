'''
Leetcode 140: Word Break II

Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.

Note that the same word in the dictionary may be reused multiple times in the segmentation.
'''
from typing import List, Set

class Solution:
    '''
    Backtracking
    - Time complexity: O(n⋅2^n)
    - Space complexity: O(2^n)
    '''
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        word_set = set(wordDict)
        results = []
        # Backtrack
        self._backtrack(s, word_set, [], results, 0)
        return results

    def _backtrack(
        self,
        s: str,
        word_set: Set[str],
        current_sentence: List[str],
        results: List[str],
        start_index: int,
        ) -> None:
        if start_index == len(s):
            results.append(" ".join(current_sentence))
            return

        # Iterate over possible end indices
        for end_index in range(start_index+1, len(s)+1):
            word = s[start_index: end_index]
            if word in word_set:
                # Make move: Add word to current sentence
                current_sentence.append(word)
                # Backtrack
                self._backtrack(s, word_set, current_sentence, results, end_index)
                # Unmake move: Remove last word to backtrack
                current_sentence.pop()