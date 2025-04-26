'''
Leetcode 212: Word Search II

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

Example 1:
Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
'''
# Mostly from https://walkccc.me/LeetCode/problems/212/#__tabbed_1_3
'''
Backtracking with Trie
 - Time complexity: O(M(4⋅3^(L−1))), where M is the number of cells in the board and L is the maximum length of words.
 - Space Complexity: O(N), where N is the total number of letters in the dictionary.
'''

from typing import Dict, List
class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end: bool = False
        self.word: str | None = None

class Trie:
    def __init__(self):
        self.root: TrieNode = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_end = True
        node.word = word


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        VISITED_CELL_CHAR = "*"
        x_dir = [1, -1, 0, 0]
        y_dir = [0, 0, 1, -1]
        # Build trie from words
        trie = Trie()
        for word in words:
            trie.insert(word)

        # Walk in the board, backtrack and find words
        m, n = len(board), len(board[0])
        result = []

        # DFS to walk on the board in cell (i,j)
        def dfs(i: int, j: int, node: TrieNode) -> None:
            # If out-of-bounds, or cell is visited, return
            if not ((0 <= i < m) and (0 <= j < n)) or \
               board[i][j] == VISITED_CELL_CHAR:
                return

            # Visit board[i][j]
            ch = board[i][j]
            # If character doesn't exist in Trie, return early
            if ch not in node.children:
                return
            # Get children of character ch in Trie
            child = node.children[ch]
            if child.word:
                result.append(child.word)
                child.word = None
            
            # Make move from board[i][j] to neighbours, while on child node of Trie
            board[i][j] = VISITED_CELL_CHAR
            # Backtrack
            for x_delta, y_delta in list(zip(x_dir, y_dir)):
                dfs(i + x_delta, j + y_delta, child)
            # Unmake move
            board[i][j] = ch

        # Start DFS from each cell of the board
        # Keep updating result inside dfs()
        for i in range(m):
            for j in range(n):
                dfs(i, j, trie.root)

        return result
