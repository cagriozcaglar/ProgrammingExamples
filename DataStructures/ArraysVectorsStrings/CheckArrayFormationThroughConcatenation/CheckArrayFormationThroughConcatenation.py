"""
You are given an array of distinct integers arr and an array of integer arrays pieces, where the integers in pieces are distinct. Your goal is to form arr by concatenating the arrays in pieces in any order. However, you are not allowed to reorder the integers in each array pieces[i].

Return true if it is possible to form the array arr from pieces. Otherwise, return false.

Note: This is a Leetcode question: https://leetcode.com/contest/weekly-contest-213/problems/check-array-formation-through-concatenation/

Example 1:
Input: arr = [85], pieces = [[85]]
Output: true

Example 2:
Input: arr = [15,88], pieces = [[88],[15]]
Output: true
Explanation: Concatenate [15] then [88]

Example 3:
Input: arr = [49,18,16], pieces = [[16,18,49]]
Output: false
Explanation: Even though the numbers match, we cannot reorder pieces[0].

Example 4:
Input: arr = [91,4,64,78], pieces = [[78],[4,64],[91]]
Output: true
Explanation: Concatenate [91] then [4,64] then [78]

Example 5:
Input: arr = [1,3,5,7], pieces = [[2,4,6,8]]
Output: false

Constraints:
    - 1 <= pieces.length <= arr.length <= 100
    - sum(pieces[i].length) == arr.length
    - 1 <= pieces[i].length <= arr.length
    - 1 <= arr[i], pieces[i][j] <= 100
    - The integers in arr are distinct.
    - The integers in pieces are distinct (i.e., If we flatten pieces in a 1D array, all the integers in this array are distinct).
"""

from typing import List


class CheckArrayFormationThroughConcatenation:
    @staticmethod
    def canFormArray(arr: List[int], pieces: List[List[int]]) -> bool:
        truth = [False] * len(arr)

        piecesIndex = 0
        arrIndex = 0
        piecesIndexMax = len(pieces) - 1
        arrIndexMax = len(arr) - 1
        # Go through array arr
        while arrIndex <= arrIndexMax:
            # print(f"Inside first while: arrIndex: {arrIndex}")
            piecesIndex = 0
            # Go through subarrays of pieces
            while piecesIndex <= piecesIndexMax:
                # print(f"Inside second while: piecesIndex: {piecesIndex}")
                # Check if subarray of arr and pieces sublist are equal
                for i in range(len(pieces[piecesIndex])):
                    # print(f"piecesIndex: {piecesIndex}, pieces[piecesIndex] : {pieces[piecesIndex]}, arrIndex : {arrIndex}")
                    if arrIndex + i > arrIndexMax:
                        break
                    if pieces[piecesIndex][i] == arr[arrIndex + i]:
                        # print(f"Found equality")
                        truth[arrIndex + i] = True
                        # arrIndex += 1
                    else:
                        # Last position matching value in pieces[piecesIndex][i]
                        arrIndex = arrIndex + i
                        break
                piecesIndex += 1
            arrIndex += 1
        # print(f"truth: {truth}")
        return all(truth)


if __name__ == "__main__":
    # Example 1:
    # Input: arr = [85], pieces = [[85]]
    # Output: true
    arr = [85]
    pieces = [[85]]
    print(f"arr: {arr}, pieces: {pieces}\nresult: {CheckArrayFormationThroughConcatenation.canFormArray(arr, pieces)}\n")

    # Example 2:
    #
    # Input: arr = [15,88], pieces = [[88],[15]]
    # Output: true
    # Explanation: Concatenate [15] then [88]
    arr = [15, 88]
    pieces = [[88], [15]]
    print(f"arr: {arr}, pieces: {pieces}\nresult: {CheckArrayFormationThroughConcatenation.canFormArray(arr, pieces)}\n")

    # Example 3:
    # Input: arr = [49,18,16], pieces = [[16,18,49]]
    # Output: false
    # Explanation: Even though the numbers match, we cannot reorder pieces[0].
    arr = [49, 18, 16]
    pieces = [[16, 18, 49]]
    print(f"arr: {arr}, pieces: {pieces}\nresult: {CheckArrayFormationThroughConcatenation.canFormArray(arr, pieces)}\n")

    # Example 4:
    # Input: arr = [91,4,64,78], pieces = [[78],[4,64],[91]]
    # Output: true
    # Explanation: Concatenate [91] then [4,64] then [78]
    arr = [91, 4, 64, 78]
    pieces = [[78], [4, 64], [91]]
    print(f"arr: {arr}, pieces: {pieces}\nresult: {CheckArrayFormationThroughConcatenation.canFormArray(arr, pieces)}\n")

    # Example 5:
    # Input: arr = [1,3,5,7], pieces = [[2,4,6,8]]
    # Output: false
    arr = [1, 3, 5, 7]
    pieces = [[2, 4, 6, 8]]
    print(f"arr: {arr}, pieces: {pieces}\nresult: {CheckArrayFormationThroughConcatenation.canFormArray(arr, pieces)}\n")
