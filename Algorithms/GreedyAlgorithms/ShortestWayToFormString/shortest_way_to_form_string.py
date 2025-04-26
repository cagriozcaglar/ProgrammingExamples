'''
Leetcode 1055: Shortest Way to Form String

From any string, we can form a subsequence of that string by deleting some number of characters (possibly no deletions).

Given two strings source and target, return the minimum number of subsequences of source such that their concatenation equals target. If the task is impossible, return -1.

Example 1:
Input: source = "abc", target = "abcbc"
Output: 2
Explanation: The target "abcbc" can be formed by "abc" and "bc", which are subsequences of source "abc".
'''
from collections import defaultdict

class Solution:
    # Solution: 2-D array keeping next occurrence of every character
    # Time complexity: O(S+T), where S and T are the lengths of source and target respectively.
    # Space complexity: O(S)
    def shortestWay(self, source: str, target: str) -> int:
        # Length of source
        source_length = len(source)

        # Next Occurrence of Character after Index
        next_occurrence = [defaultdict(int) for idx in range(source_length)]

        # Base Case
        next_occurrence[source_length - 1][source[source_length - 1]] = source_length - 1

        # Using Recurrence Relation to fill next_occurrence
        for idx in range(source_length - 2, -1, -1):
            next_occurrence[idx] = next_occurrence[idx + 1].copy()
            next_occurrence[idx][source[idx]] = idx

        # Pointer for source
        source_iterator = 0

        # Number of times we need to iterate through source
        count = 1

        # Find all characters of target in source
        for char in target:

            # If character is not in source, return -1
            if char not in next_occurrence[0]:
                return -1

            # If we have reached the end of source, or the character is not in
            # source after source_iterator, loop back to beginning
            if (source_iterator == source_length or
                    char not in next_occurrence[source_iterator]):
                count += 1
                source_iterator = 0

            # Next occurrence of character in source after source_iterator
            source_iterator = next_occurrence[source_iterator][char] + 1

        # Return the number of times we need to iterate through source
        return count

    # Solution: Two-pointers
    # Time complexity: O(S⋅T), where T is length of target and S is length of source
    # Space complexity: O(1)
    def shortestWay(self, source: str, target: str) -> int:
        # Check if all chars in target exist in source. Otherwise, return -1
        source_chars = set(source)
        for char in target:
            if char not in source_chars:
                return -1

        # Length of source to loop back to start of start using mod
        m = len(source)
        # Source Iterator
        s = 0
        # Number of times source is traversed, which equals min_num_subseq
        min_num_subseq = 0

        # Iterate over target, find subsequences of source, increment count
        for char in target:
            # If s = 0, starting a new subsequence, increment count
            if s == 0:
                min_num_subseq += 1

            # Find first occurrence of char in source
            while source[s] != char:
                # Increment pointer s, using modulo m, to return to start
                s = (s + 1) % m
                # If s = 0, starting a new subsequence, increment count
                if s == 0:
                    min_num_subseq += 1

            # Loop will break, when char is found in source. Thus, increment.
            # Don't increment count until it is not clear that target has chars left.
            s = (s + 1) % m

        return min_num_subseq
