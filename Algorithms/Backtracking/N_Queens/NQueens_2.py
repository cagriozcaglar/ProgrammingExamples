'''
Leetcode 52: N-Queens II
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
Given an integer n, return the number of distinct solutions to the n-queens puzzle.
'''
# Time complexity: O(N!)
# Space complexity: O(N^2)
class Solution:
    def totalNQueens(self, n: int) -> int:
        def backtrack(row, cols, diagonals, anti_diagonals):
            # Base case - N queens have been placed, row == n, return 1
            if row == n:
                return 1

            num_solutions = 0
            # Construct candidates
            # Check if new row conflicts with column, diagonal, and anti-diagonal
            for col in range(n):
                curr_diagonal = row - col
                curr_anti_diagonal = row + col
                # If queen is not place-able
                if(col in cols or
                    curr_diagonal in diagonals or
                    curr_anti_diagonal in anti_diagonals):
                   continue

                # Make move: Add queen to the board
                cols.add(col)
                diagonals.add(curr_diagonal)
                anti_diagonals.add(curr_anti_diagonal)

                # Backtrack: Move to next row with updated board state
                num_solutions += backtrack(row + 1, cols, diagonals, anti_diagonals)

                # Unmake move: Remove queen from the board since we already explored
                # all valid paths using the above function call.
                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)
            return num_solutions

        return backtrack(0, set(), set(), set())