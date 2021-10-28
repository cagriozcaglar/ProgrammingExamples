"""
Left-most column with one
Binary Search, or linear pass
"""
# """
# This is BinaryMatrix's API interface.
# You should not implement it, or speculate about its implementation
# """
from typing import List
class BinaryMatrix(object):
   def get(self, row: int, col: int) -> int:
       pass
   def dimensions(self) -> List[int]:
       pass

class Solution:
    # Linear pass on the matrix: O(m+n) (n: #rows, m = #cols)
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        rows, cols = binaryMatrix.dimensions()
        # Pointers
        currentRow = 0
        currentCol = cols - 1

        # Repeat the search until it goes off the grid
        while currentRow < rows and currentCol >= 0:
            # If 0, go down, increment row
            if binaryMatrix.get(currentRow, currentCol) == 0:
                currentRow += 1
            # else, if 1, fo left, decrement col
            else:
                currentCol -= 1

        return currentCol+1 if currentCol != cols-1 else -1

    # Binary Search on rows: O(n * log(m)) (n: #rows, m = #cols)
    def leftMostColumnWithOneBinarySearch(self, binaryMatrix: 'BinaryMatrix') -> int:
        rows, cols = binaryMatrix.dimensions()
        smallestIndex = cols  # max value (> cols indices)

        # For each row, use binary search, find smallest index
        for row in range(rows):
            # Binary Search for the first 1 in the row
            low = 0
            high = cols-1
            while low < high:
                mid = (low+high) // 2
                if binaryMatrix.get(row, mid) == 0:  # right
                    low = mid+1
                else: # left
                    high = mid
            # If last element in search space is a 1
            if binaryMatrix.get(row, low) == 1:
                smallestIndex = min(smallestIndex, low)
        # return smallestIndex (or -1 if all zeros: smallestIndex == cols)
        return -1 if smallestIndex==cols else smallestIndex