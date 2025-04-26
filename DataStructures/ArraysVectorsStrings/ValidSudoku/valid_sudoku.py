'''
Leetcode 36: Valid Sudoku

Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
Note:

A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
'''
from typing import List

'''
Algorithm

1. Initialize a list containing 9 hash sets, where the hash set at index r will be used to store previously seen numbers in row r of the sudoku. Likewise, initialize lists of 9 hash sets to track the columns and boxes too.
2. Iterate over each position (r, c) in the sudoku. At each iteration, if there is a number at the current position:
  - Check if the number exists in the hash set for the current row, column, or box. If it does, return false, because this is the second occurrence of the number in the current row, column, or box.
  - Otherwise, update the set responsible for tracking previously seen numbers in the current row, column, and box. The index of the current box is (r / 3) * 3 + (c / 3) where / represents floor division.
3. If no duplicates were found after every position on the sudoku board has been visited, then the sudoku is valid, so return true.

- Time complexity: O(N^2) because we need to traverse every position in the board, and each of the four check steps is an O(1) operation.
- Space complexity: O(N^2) because in the worst-case scenario, if the board is full, we need a hash set each with size N to store all seen numbers for each of the N rows, N columns, and N boxes, respectively.
'''
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        n = 9
        
        rows = [set() for _ in range(n)]
        cols = [set() for _ in range(n)]
        boxes = [set() for _ in range(n)]

        for row in range(len(board)):
            for col in range(len(board[0])):
                val = board[row][col]
                if val == '.':
                    continue
                    
                # Row check
                if val in rows[row]:
                    return False
                rows[row].add(val)

                # Column check
                if val in cols[col]:
                    return False
                cols[col].add(val)
                
                # Box check
                box_index = (row // 3) * 3 + (col // 3)
                if val in boxes[box_index]:
                    return False
                boxes[box_index].add(val)
        
        return True
