'''
Leetcode 737: Sentence Similarity II

We can represent a sentence as an array of words, for example, the sentence "I am happy with leetcode" can be represented as arr = ["I","am",happy","with","leetcode"].

Given two sentences sentence1 and sentence2 each represented as a string array and given an array of string pairs similarPairs where similarPairs[i] = [xi, yi] indicates that the two words xi and yi are similar.

Return true if sentence1 and sentence2 are similar, or false if they are not similar.

Two sentences are similar if:

They have the same length (i.e., the same number of words)
sentence1[i] and sentence2[i] are similar.
Notice that a word is always similar to itself, also notice that the similarity relation is transitive. For example, if the words a and b are similar, and the words b and c are similar, then a and c are similar.

Example 1:
Input: sentence1 = ["great","acting","skills"], sentence2 = ["fine","drama","talent"], similarPairs = [["great","good"],["fine","good"],["drama","acting"],["skills","talent"]]
Output: true
Explanation: The two sentences have the same length and each word i of sentence1 is also similar to the corresponding word in sentence2.

Example 2:
Input: sentence1 = ["I","love","leetcode"], sentence2 = ["I","love","onepiece"], similarPairs = [["manga","onepiece"],["platform","anime"],["leetcode","platform"],["anime","manga"]]
Output: true
Explanation: "leetcode" --> "platform" --> "anime" --> "manga" --> "onepiece".
Since "leetcode is similar to "onepiece" and the first two words are the same, the two sentences are similar.
'''

from typing import Dict, List
from collections import defaultdict

class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = defaultdict(str)
        self.rank: Dict[str, int] = defaultdict(int)

    def add_word(self, word: str) -> None:
        if word not in self.parent:
            self.parent[word] = word
            self.rank[word] = 0

    def is_word_present(self, word: str) -> bool:
        return word in self.parent

    def find(self, word: str) -> str:
        if self.parent[word] != word:
            self.parent[word] = self.find(self.parent[word])
        return self.parent[word]

    def union(self, word1: str, word2: str) -> None:
        root1, root2 = self.find(word1), self.find(word2)
        if root1 == root2:  # Same group, no need to union
            return
        elif self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:  # ranks are equal, make root1 parent ot root2
            self.parent[root2] = root1
            self.rank[root1] = self.rank[root1] + 1

class Solution:
    def areSentencesSimilarTwo(self, sentence1: List[str], sentence2: List[str], similarPairs: List[List[str]]) -> bool:
        # If sent1 & sent lengths are not equal, return False
        if len(sentence1) != len(sentence2):
            return False

        # UnionFind DS to check if two words are similar, in O(1) time
        uf: UnionFind = UnionFind()
        for pair in similarPairs:
            uf.add_word(pair[0])
            uf.add_word(pair[1])
            uf.union(pair[0], pair[1])

        # Iterate over the sentence pair words and check if all words are similar
        for i in range(len(sentence1)):
            if sentence1[i] == sentence2[i]:
                continue
            if uf.is_word_present(sentence1[i]) and \
               uf.is_word_present(sentence2[i]) and \
               uf.find(sentence1[i]) == uf.find(sentence2[i]):
               continue
            return False

        return True