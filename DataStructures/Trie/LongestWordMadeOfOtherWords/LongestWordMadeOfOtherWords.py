"""
Longest word made of other words
Given an array of strings words representing an English Dictionary, return the longest word in words that can be built
one character at a time by other words in words.
"""
# https://leetcode.com/problems/longest-word-in-dictionary/discuss/113916/Python%3A-Trie-%2B-BFS
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.isEnd = False
        self.word = ""

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.isEnd = True
        node.word = word

    def bfs(self):
        bfsQueue = collections.deque([self.root])
        result = ""
        while bfsQueue:
            cur = bfsQueue.popleft()
            for node in cur.children.values():
                if node.isEnd:
                    bfsQueue.append(node)
                    if len(node.word) > len(result) or node.word < result:
                        result = node.word
        return result

class Solution:

    """
    Trie + BFS: O(sum_i w_i), w_i is length of i-th word
    """
    def longestWord(self, words: List[str]) -> str:
        trie = Trie()
        for word in words:
            trie.insert(word)
        return trie.bfs()

    """
    Brute Force: O(sum_i w_i^2), w_i is length of i-th word
    Checking all prefixes of w_i.
    """
    def longestWordBruteForce(self, words: List[str]) -> str:
        wordSet = set(words)
        # Sort by decreasing length first, then increasing chrono-order
        words.sort(key = lambda word: (-len(word), word))
        for word in words:
            if all(word[:k] in wordSet for k in range(1, len(word))):
                return word

        return ""