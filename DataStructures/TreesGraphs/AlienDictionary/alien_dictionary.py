'''
Leetcode 269: Alien Dictionary

There is a new alien language that uses the English alphabet. However, the order among the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary, where the strings in words are sorted lexicographically by the rules of this new language.

Return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there is no solution, return "". If there are multiple solutions, return any of them.

A string s is lexicographically smaller than a string t if at the first letter where they differ, the letter in s comes before the letter in t in the alien language. If the first min(s.length, t.length) letters are the same, then s is smaller if and only if s.length < t.length.

Example 1:
Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"

Example 2:
Input: words = ["z","x"]
Output: "zx"
'''
from typing import List, Dict, Set
from collections import defaultdict

# Myself, plus https://neetcode.io/problems/foreign-dictionary
class Solution:
    '''
    1. Iterate over word pairs in words, and find character orderings
      a. Create adjacency list based on these character orderings. Graph G.
    2. Run Topological sort on Graph G.
      a. Run DFS on G.
        i. If there is a cycle, no such ordering exists, return ""
        ii. If no cycle, after all characters are visited, sort them by
            decreasing order of finish times.
    '''
    def alienOrder(self, words: List[str]) -> str:
        # 1. Iterate over word pairs in words, and find character orderings
        # a. Create adjacency list based on these character orderings. Graph G.
        adj_list: Dict[str, Set[str]] = {ch: set() for w in words for ch in w}

        for i in range(len(words)-1):
            word1, word2 = words[i], words[i+1]
            min_len = min(len(word1), len(word2))
            # Edge case: word2 is unequal prefix of word1. IMPORTANT: You forgot this edge case
            if len(word1) > len(word2) and word1[:min_len] == word2[:min_len]:
                return ""
            for i in range(min_len):
                if word1[i] != word2[i]:
                    adj_list[word1[i]].add(word2[i])
                    break

        # 2. Run Topological sort on Graph G.
        # a. Run DFS on G.
        #     i. If there is a cycle, no such ordering exists, return ""
        #     ii. If no cycle, after all characters are visited, sort them by
        #         decreasing order of finish times.

        # Visited map: False = grey, True = black
        visited_map: Dict[str, bool] = defaultdict(bool)
        output = []

        def dfs_visit(node: str) -> bool:
            if node in visited_map:
                # If this node is grey (False), a cycle is detected
                return visited_map[node]

            visited_map[node] = False  # Exploring node

            for nei in adj_list[node]:
                if not dfs_visit(nei):
                    return False  # Cycle was detected down the DFS tree

            visited_map[node] = True  # Mark node as visited / black
            output.append(node)
            return True

        for ch in adj_list:
            if not dfs_visit(ch):
                return ""

        # Topological sort, in reverse order of finishing times of nodes
        return "".join(output[::-1])