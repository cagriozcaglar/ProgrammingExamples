'''
Leetcode 1698: Number of Distinct Substrings in a String

Given a string s, return the number of distinct substrings of s.

A substring of a string is obtained by deleting any number of characters (possibly zero)
from the front of the string and any number (possibly zero) from the back of the string.

Example 1:
Input: s = "aabbaba"
Output: 21
Explanation: The set of distinct strings is ["a","b","aa","bb","ab","ba","aab","abb","bab",
"bba","aba","aabb","abba","bbab","baba","aabba","abbab","bbaba","aabbab","abbaba","aabbaba"]

Example 2:
Input: s = "abcdefg"
Output: 28
'''

# Solution from https://leetcode.ca/2020-07-24-1698-Number-of-Distinct-Substrings-in-a-String/
# Explanation: Build a trie of suffixes (suffix tree), and each internal node of the tree
# represents a distinct prefix of a suffix, which is a distinct substring of the string.

# Use a Trie, and every time a new Trie node created, meaning a new substring.
# More on complexity analysis of Trie solution:
# 1. The Trie construction involves iterating over all suffixes of the string and, for each suffix,
# possibly traversing and inserting characters into the Trie. The number of operations is tied to the
# total length of all suffixes, which in theory gives a complexity of (O(n^2)) for string length (n).
# 2. However, due to the efficient nature of Trie operations (where each character insertion/check is
# (O(1)) assuming a fixed character set), and the fact that many common substrings in the suffixes do
# not lead to repeated insertions after the first occurrence, the practical performance approaches (O(n))
# for inserting all characters of the string into the Trie.
# 3. Note that the theoretical worst-case time complexity might not strictly be (O(n)), but the Trie
# approach significantly reduces redundant comparisons between substrings, making it highly efficient for
# this problem.

from collections import defaultdict
from typing import Dict

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.is_end: bool = False

class Trie:
    def __init__(self):
        self.root: TrieNode = TrieNode()

class Solution:
    def countDistinct(self, s: str) -> int:
        # Initiate a new trie, initialize count to 0
        trie = Trie()
        count = 0

        # For each starting point i of a suffix, start from trie root
        for i in range(len(s)):
            current = trie.root
            # End index j of suffix
            for j in range(i, len(s)):
                # if s[j] not in children, add new node
                # Increment count for each new node in the trie
                # Each new node represents a distinct substring (prefix of a suffix)
                if s[j] not in current.children:
                    current.children[s[j]] = TrieNode()
                    count += 1
                current = current.children[s[j]]

        return count
