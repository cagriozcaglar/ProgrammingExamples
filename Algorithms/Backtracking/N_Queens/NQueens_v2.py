'''
Leetcode 51:
The n-queens puzzle is the problem of placing n queens on an n x n chessboard such that no two queens attack each other.
Given an integer n, return all distinct solutions to the n-queens puzzle. You may return the answer in any order.
Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space, respectively.
'''
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        # Write solution from state
        def create_solution_from_state(state):
            board = []
            for row in state:
                board.append("".join(row))  # Append row
            return board

        def backtrack(row, cols, diagonals, anti_diagonals, state):
            # Base case - N queens have been placed, row == n
            if row == n:
                solutions.append(create_solution_from_state(state))
                return
            
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
                state[row][col] = 'Q'

                # Backtrack: Move to next row with updated board state
                backtrack(row + 1, cols, diagonals, anti_diagonals, state)

                # Unmake move: Remove queen from the board since we already explored
                # all valid paths using the above function call.
                cols.remove(col)
                diagonals.remove(curr_diagonal)
                anti_diagonals.remove(curr_anti_diagonal)
                state[row][col] = '.'

        solutions = []
        # Empty nxn board with .'s only
        empty_board = [["."] * n for _ in range(n)]
        backtrack(0, set(), set(), set(), empty_board)
        return solutions
