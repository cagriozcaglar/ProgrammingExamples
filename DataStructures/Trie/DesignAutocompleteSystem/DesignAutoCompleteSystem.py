'''
Leetcode 642: Design Search Autocomplete System

Design a search autocomplete system for a search engine. Users may input a sentence (at least one word and end with a special character '#').

You are given a string array sentences and an integer array times both of length n where sentences[i] is a previously typed sentence and times[i] is the corresponding number of times the sentence was typed. For each input character except '#', return the top 3 historical hot sentences that have the same prefix as the part of the sentence already typed.

Here are the specific rules:

The hot degree for a sentence is defined as the number of times a user typed the exactly same sentence before.
The returned top 3 hot sentences should be sorted by hot degree (The first is the hottest one). If several sentences have the same hot degree, use ASCII-code order (smaller one appears first).
If less than 3 hot sentences exist, return as many as you can.
When the input is a special character, it means the sentence ends, and in this case, you need to return an empty list.

Implement the AutocompleteSystem class:
AutocompleteSystem(String[] sentences, int[] times) Initializes the object with the sentences and times arrays.
List<String> input(char c) Returns the top 3 historical hot sentences that have a common prefix with the sentence currently being typed. If there are fewer than 3 matches, return them all.
'''

from typing import Dict, List
from collections import defaultdict
import heapq

# Solution: Trie + Heap
class TrieNode:
    def __init__(self):
        self.children = {}
        # sentences: Map from complete sentence (with the prefix) to its hot degree
        self.sentences = defaultdict(int)

class AutocompleteSystem:

    def __init__(self, sentences: List[str], times: List[int]):
        self.root = TrieNode()
        for sentence, count in zip(sentences, times):
            self.add_to_trie(sentence, count)
        self.curr_sentence = []
        self.curr_node = self.root
        self.dead = TrieNode()

    # Internal insert method for Trie, can be moved out to a Trie class
    def add_to_trie(self, sentence, count):
        node = self.root
        for c in sentence:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.sentences[sentence] -= count


    def input(self, c: str) -> List[str]:
        # Case 1: c == '#': We have finished typing the current sentence.
        if c == "#":
            curr_sentence = "".join(self.curr_sentence)
            self.add_to_trie(curr_sentence, 1)
            self.curr_sentence = []
            self.curr_node = self.root
            return []

        # Case 2: c != '#', but c is not a child of currNode, return []
        self.curr_sentence.append(c)
        if c not in self.curr_node.children:
            self.curr_node = self.dead
            return []

        # Case 3: c != '#', and c is a child of currNode, return top 3
        self.curr_node = self.curr_node.children[c]
        items = [(val, key) for key, val in self.curr_node.sentences.items()]
        answer = heapq.nsmallest(3, items)
        return [item[1] for item in answer]


# Your AutocompleteSystem object will be instantiated and called as such:
# obj = AutocompleteSystem(sentences, times)
# param_1 = obj.input(c)



# class TrieNode:
#     def __init__(self):
#         self.node: Dict[str, TrieNode] = defaultdict(TrieNode)
#         self.isEnd: bool = False
#         self.hot: int = 0

# class AutocompleteSystem:
#     def __init__(self, sentences: List[str], times: List[int]):
#         self.root: TrieNode = TrieNode()
#         self.searchTerm = ""
#         # Populate trie with (sentence, times) pairs using insert() method
#         for index, sentence in enumerate(sentences):
#             self.insert(sentence, times[index])

#     def insert(self, word: str, hot: int) -> None:
#         current: TrieNode = self.root
#         for char in word:
#             current = current.node[char]
#         current.isEnd = True
#         # Negate hot field, because we can sort in ascending (increasing) order later to get top 3
#         current.hot -= hot

#     def search(self) -> List[str]:
#         current: TrieNode = self.root
#         result = []
#         path = ""
#         for char in self.searchTerm:
#             if char not in current.node:
#                 return result
#             path += char
#             current = current.node[char]
#         self.dfs(current, path, result)
#         return [item[1] for item in sorted(result)[:3]]

#     def dfs(self, node: TrieNode, word: str, result: List[str]) -> None:
#         if node.isEnd:
#             result.append( (node.hot, word) )
#         for char, newNode in node.node.items():
#             self.dfs(newNode, word+char, result)

#     def input(self, c: str) -> List[str]:
#         if c != "#":
#             self.searchTerm += c
#             return self.search()
#         else:
#             self.insert(self.searchTerm, 1)
#             self.searchTerm = ""
