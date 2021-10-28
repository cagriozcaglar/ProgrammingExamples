# Perfect (Python): https://www.programmersought.com/article/62487348507/
# OK (Java): https://walkccc.me/LeetCode/problems/0773/
"""
Think of the state of the board as the nodes of the graph, and the transition between them as edges. In fact, this is the problem of finding the shortest distance between nodes, just use the BFS template.

Several points to note:

1) Because the list is unhashable, it cannot be added to the visited set, so some conversions between the list and the string are involved here
2) The position correspondence of an element in the list in the flattened string. Assuming that the position in the list is [x,y], the position in the corresponding string is x*num_col+y. If the position in the string is k, then the corresponding position in the list is [k//num_col,k% num_col]
3) Also note that when traversing in four directions, the board needs to backtrack back when switching to a new direction.
4) Also pay attention to judging the boundary and judging whether the location in the code has been visited. Generally speaking, these two judgments are together, but because some conversions between list and string are involved, these two need to be separated
"""
"""
Sliding Puzzle
"""
import collections
from typing import List

class Solution:
    # BFS + Backtracking (BFS => Greedy, because we need min number of moves)
    # Shortest path problem, but, because edges are unweighted, it can be solved with BFS
    def slidingPuzzle(self, board: List[List[int]]) -> int:
        # Global hard-coded variables
        numRows = 2
        numCols = 3

        # Convert serialized string index to board index
        def convertToBoardIndex(stringIndex: int) -> List[int]:
            return [stringIndex // numCols, stringIndex % numCols]

        def inBounds(x: int, y: int) -> bool:
            return x >= 0 and x < numRows and y >= 0 and y < numCols

        # Serialized goal state
        goal = "123450"

        # Serialized start state
        start = ""
        for i in range(numRows):
            for j in range(numCols):
                start += str(board[i][j])

        # Graph: vertices are board states
        # Start BFS queue, push start state to queue, along with #movesSoFar
        q = collections.deque()
        q.append((start,0))

        # Keep visited set of vertices (board states)
        visited = set()
        visited.add(start)

        # Possible moves in 4 directions
        directions = [
            [0, 1],
            [0, -1],
            [1, 0],
            [-1, 0]
        ]

        while q:
            state, movesSoFar = q.popleft()
            # 1. If this is a solution
            if state == goal:
                # 2. Process solution
                return movesSoFar
            # Get the index of '0' (empty square)
            pos = state.index('0')
            row, col = convertToBoardIndex(pos)
            state = list(state)
            # 3. Construct candidates
            for d in directions:
                x = row + d[0]
                y = col + d[1]
                if not inBounds(x, y):
                    continue
                # 4. Make move
                # Swap board(row*numCols+col) with board(x*numCols+y)
                state[x*numCols+y], state[row*numCols+col] = state[row*numCols+col], state[x*numCols+y]
                stateStr = "".join(state)
                if stateStr not in visited:
                    q.append( (stateStr, movesSoFar+1) )
                    visited.add(stateStr)
                # 6. Unmake move
                # Swap board(row*numCols+col) with board(x*numCols+y)
                state[x*numCols+y], state[row*numCols+col] = state[row*numCols+col], state[x*numCols+y]
        return -1