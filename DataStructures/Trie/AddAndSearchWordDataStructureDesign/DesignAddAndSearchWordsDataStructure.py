"""
Design Add and Search Words Data Structure
"""
from typing import Dict
from collections import defaultdict
class TrieNode:
    def __init__(self):
        self.links: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.isLeaf: bool = False

class WordDictionary:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for char in word:
            node = node.links[char]
        node.isLeaf = True

    def search(self, word: str) -> bool:
        def dfs(node, i) -> bool:
            if i == len(word):
                return node.isLeaf
            # Handling "." character via recursive dfs() call
            if word[i] == ".":
                for child in node.links:
                    if dfs(node.links[child], i+1):
                        return True
            # Next char is available, call dfs() recursively
            if word[i] in node.links:
                return dfs(node.links[word[i]], i+1)
            # Otherwise, character is not "." or char is not in list, return False
            return False

        return dfs(self.root, 0)

    def searchBasic(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.links:
                return False
            node = node.links[char]
        return node.isLeaf

# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)