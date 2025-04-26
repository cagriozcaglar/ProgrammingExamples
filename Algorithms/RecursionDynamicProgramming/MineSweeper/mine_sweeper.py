'''
Leetcode 529: Mine Sweeper

Let's play the minesweeper game (Wikipedia, online game)!

You are given an m x n char matrix board representing the game board where:

'M' represents an unrevealed mine,
'E' represents an unrevealed empty square,
'B' represents a revealed blank square that has no adjacent mines (i.e., above, below, left, right, and all 4 diagonals),
digit ('1' to '8') represents how many mines are adjacent to this revealed square, and
'X' represents a revealed mine.

You are also given an integer array click where click = [clickr, clickc] represents the next click position among all the unrevealed squares ('M' or 'E').

Return the board after revealing this position according to the following rules:

If a mine 'M' is revealed, then the game is over. You should change it to 'X'.
If an empty square 'E' with no adjacent mines is revealed, then change it to a revealed blank 'B' and all of its adjacent unrevealed squares should be revealed recursively.
If an empty square 'E' with at least one adjacent mine is revealed, then change it to a digit ('1' to '8') representing the number of adjacent mines.
Return the board when no more squares will be revealed.

Example 1:
Input: board = [["E","E","E","E","E"],["E","E","M","E","E"],["E","E","E","E","E"],["E","E","E","E","E"]], click = [3,0]
Output: [["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]

Example 2:
Input: board = [["B","1","E","1","B"],["B","1","M","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]], click = [1,2]
Output: [["B","1","E","1","B"],["B","1","X","1","B"],["B","1","1","1","B"],["B","B","B","B","B"]]
'''
from typing import List

# Solution is from https://leetcode.com/problems/minesweeper/discuss/99897/10-line-python-solution
class Solution:
    def updateBoard(self, board: List[List[str]], click: List[int]) -> List[List[str]]:
        def isWithinBounds(row: int, col: int) -> bool:
            return 0 <= row < len(board) and 0 <= col < len(board[0])

        (row, col), directions = click, [
            [-1, 0],
            [1, 0],
            [0, 1],
            [0, -1],
            [-1, 1],
            [-1, -1], 
            [1, 1],
            [1, -1]
        ]
        if isWithinBounds(row, col):
            if board[row][col] == "M":
                board[row][col] = "X"
            elif board[row][col] == "E":
                n = sum([ board[row+r][col+c] == "M" for r,c in directions if isWithinBounds(row+r, col+c)])
                if n > 0:
                    board[row][col] = str(n)
                else:
                    board[row][col] = "B"
                    for r, c in directions:
                        self.updateBoard(board, [row+r, col+c])
        return board