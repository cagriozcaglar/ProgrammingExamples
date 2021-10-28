# https://stackoverflow.com/questions/46038694/implementing-a-trie-to-support-autocomplete-in-python
from typing import Dict
from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.node: Dict[str, TrieNode] = defaultdict(TrieNode)
        self.isEnd: bool = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        current: TrieNode = self.root
        for char in word:
            current = current.node[char]
        current.isEnd = True

    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        current: TrieNode = self.root
        for char in word:
            if char not in current.node:
                return False
            current = current.node[char]
        return current.isEnd

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        current: TrieNode = self.root
        for char in prefix:
            if char not in current.node:
                return False
            current = current.node[char]
        return True

    def dfs(self, root: TrieNode, word: str, wordList: List[str]) -> None:
        if root.isEnd:
            wordList.append(word)
        for char, newNode in root.node.items():
            self.dfs(newNode, word+char)

    def autoComplete(self, searchWord: str, wordList: List[str]) -> None:
        tempWord = ""
        current = self.root
        for char in searchWord:
            if char not in current.node.keys():
                print("Invalid input")
            else:
                tempWord += char
                current = current.node[char]
        self.dfs(current, tempWord, wordList)
        print(wordList)


# Your Trie object will be instantiated and called as such:
# obj = Trie()
# obj.insert(word)
# param_2 = obj.search(word)
# param_3 = obj.startsWith(prefix)