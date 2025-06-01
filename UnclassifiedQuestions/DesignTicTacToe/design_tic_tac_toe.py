'''
Leetcode 348: Design Tic-Tac-Toe

Assume the following rules are for the tic-tac-toe game on an n x n board between two players:

A move is guaranteed to be valid and is placed on an empty block.
Once a winning condition is reached, no more moves are allowed.
A player who succeeds in placing n of their marks in a horizontal, vertical, or diagonal row wins the game.
Implement the TicTacToe class:

TicTacToe(int n) Initializes the object the size of the board n.
int move(int row, int col, int player) Indicates that the player with id player plays at the cell (row, col) of the board. The move is guaranteed to be a valid move, and the two players alternate in making moves. Return
0 if there is no winner after the move,
1 if player 1 is the winner after the move, or
2 if player 2 is the winner after the move.
'''

class TicTacToe:

    def __init__(self, n: int):
        self.n = n
        self.rows = [0] * n
        self.cols = [0] * n
        self.diag = 0
        self.antidiag = 0


    def move(self, row: int, col: int, player: int) -> int:
        current_player = 1 if (player == 1) else -1

        self.rows[row] += current_player
        self.cols[col] += current_player

        # Update diag
        if row == col:
            self.diag += current_player

        # Update anti-diag
        if col == self.n - row - 1:
            self.antidiag += current_player

        # Check if current player wins
        if(
            abs(self.rows[row]) == self.n or \
            abs(self.cols[col]) == self.n or \
            abs(self.diag) == self.n or \
            abs(self.antidiag) == self.n \
        ):
            return player

        # No one wins
        return 0