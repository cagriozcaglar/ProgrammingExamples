'''
Leetcode 472: Concatenated Words

Given an array of strings words (without duplicates), return all the concatenated words in the given list of words.

A concatenated word is defined as a string that is comprised entirely of at least two shorter words (not necessarily distinct) in the given array.

Example 1:
Input: words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
"dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".

Example 2:
Input: words = ["cat","dog","catdog"]
Output: ["catdog"]
'''
from typing import List

class Trie:
    def __init__(self):
        self.children = [None] * 26  # Initialize the trie nodes for each letter of the alphabet
        self.is_end_of_word = False   # Boolean to check if the node is the end of a word

    def insert(self, word):
        node = self
        for char in word:
            index = ord(char) - ord('a')  # Calculate the alphabetical index
            # If the character doesn't have an associated Trie node, create it
            if node.children[index] is None:
                node.children[index] = Trie()
            # Move to the child node associated with the character
            node = node.children[index]
        # Mark the last node as the end of a word
        node.is_end_of_word = True


class Solution:
    def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:

        # Helper function to perform depth-first search in the trie
        def dfs(word):
            if not word:  # If the word is empty, we've found a valid concatenation
                return True
            node = trie  # Start at the root of the trie
            for i, char in enumerate(word):
                index = ord(char) - ord('a')
                # If the character is not in the trie, the word cannot be formed
                if node.children[index] is None:
                    return False
                node = node.children[index]
                # If the current node is an end of a word and the next substring is also a word
                if node.is_end_of_word and dfs(word[i + 1:]):
                    return True  # The word can be concatenated from other words
            return False  # No valid concatenation found

        # Initialize Trie and the answer list
        trie = Trie()
        concatenated_words = []
        # Sort the words by length to ensure that shorter words are inserted into the trie first
        words.sort(key=len)
        for word in words:
            # Check if the word can be formed by the concatenation of other words
            if dfs(word):
                concatenated_words.append(word)
            else:
                # If not, insert the word into the trie for future reference
                trie.insert(word)
        return concatenated_words