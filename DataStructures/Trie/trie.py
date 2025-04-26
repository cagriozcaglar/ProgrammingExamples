'''
Trie implementation from https://leetcode.com/problems/implement-trie-prefix-tree/ and 
https://stackoverflow.com/questions/46038694/implementing-a-trie-to-support-autocomplete-in-python
'''

from typing import Dict
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.is_end: bool = False

class Trie:
    def __init__(self):
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children[ch]
        node.is_end = True

    def search(self, word: str) -> bool:
        (result, node) = self.startsWithResultWithNode(word)
        return result and node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True

    def startsWithResultWithNode(self, prefix: str) -> tuple[bool, TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return (False, node)
            node = node.children[ch]
        return (True, node)

# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)