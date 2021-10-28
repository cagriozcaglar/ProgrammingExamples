from typing import Dict, List
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.node: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.isEnd: bool = False
        self.hot: int = 0

class AutocompleteSystem:
    def __init__(self, sentences: List[str], times: List[int]):
        self.root: TrieNode = TrieNode()
        self.searchTerm = ""
        # Populate trie with (sentence, times) pairs using insert() method
        for index, sentence in enumerate(sentences):
            self.insert(sentence, times[index])

    def insert(self, word: str, hot: int) -> None:
        current: TrieNode = self.root
        for char in word:
            current = current.node[char]
        current.isEnd = True
        # Negate hot field, because we can sort in ascending (increasing) order later to get top 3
        current.hot -= hot

    def search(self) -> List[str]:
        current: TrieNode = self.root
        result = []
        path = ""
        for char in self.searchTerm:
            if char not in current.node:
                return result
            path += char
            current = current.node[char]
        self.dfs(current, path, result)
        return [item[1] for item in sorted(result)[:3]]

    def dfs(self, node: TrieNode, word: str, result: List[str]) -> None:
        if node.isEnd:
            result.append( (node.hot, word) )
        for char, newNode in node.node.items():
            self.dfs(newNode, word+char, result)

    def input(self, c: str) -> List[str]:
        if c != "#":
            self.searchTerm += c
            return self.search()
        else:
            self.insert(self.searchTerm, 1)
            self.searchTerm = ""
