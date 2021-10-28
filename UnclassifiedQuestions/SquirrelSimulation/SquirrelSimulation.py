"""
Note: This is Leetcode problem 573: https://leetcode.com/problems/squirrel-simulation/

There's a tree, a squirrel, and several nuts. Positions are represented by the cells in a 2D grid.
Your goal is to find the minimal distance for the squirrel to collect all the nuts and put them under the tree one by
one. The squirrel can only take at most one nut at one time and can move in four directions - up, down, left and right,
to the adjacent cell. The distance is represented by the number of moves.

Example 1:
Input:
Height : 5
Width : 7
Tree position : [2,2]
Squirrel : [4,4]
Nuts : [[3,0], [2,5]]
Output: 12
Explanation:
    Squirrel ([4,4]) -> Nut ([2,5]) -> Tree ([2,2]) -> Nut ([3,0]) -> Tree ([2,2])
                     3       +       3      +        3      +      3                =   12
Note:
 - All given positions won't overlap.
 - The squirrel can take at most one nut at one time.
 - The given positions of nuts have no order.
 - Height and width are positive integers. 3 <= height * width <= 10,000.
- The given positions contain at least one nut, only one tree and one squirrel.
"""

from typing import List


class SquirrelSimulation:
    @staticmethod
    def minDistance(height: int, width: int, tree: List[int], squirrel: List[int], nuts: List[List[int]]) -> int:
        if len(nuts) == 0:
            return 0
        totalDistance = 0
        # While traversing over the nuts array and adding the to-and-fro distance, we find out the saving, dd, which can
        # be obtained if the squirrel goes to the current nut first. Out of all the nuts, we find out the nut which
        # maximizes the saving and then deduct this maximum saving from the sum total of the to-and-fro distance of all
        # the nuts.
        # Note that the first nut to be picked needs not necessarily be the nut closest to the squirrel's start point,
        # but it's the one which maximizes the savings.
        # First nut: Minimize the distance difference:
        #    dist(sq, nut) + dist(nut, tree) // sq -> nut -> tree for first nut
        #    - 2*dist(nut, tree)             // tree -> nut -> tree for other
        #   = dist(sq, nut) - dist(nut, tree)
        # First nut = nut* = argmin_{nut \in nuts} dist(sq, nut) - dist(nut, tree)
        # For other nuts = totalDistance = 2*dist(nut, tree)
        nutDistDifferences = [distance(squirrel, nut) - distance(nut, tree) for nut in nuts]
        minNutDistDifference = min(nutDistDifferences)
        minNutDistDifferenceIndex = nutDistDifferences.index(minNutDistDifference)
        minNut = nuts[minNutDistDifferenceIndex]
        # Total distance for the first nut: sq -> nut -> tree
        totalDistance = distance(squirrel, minNut) + distance(minNut, tree)

        # Rest of the nuts
        nuts.pop(minNutDistDifferenceIndex)
        nutToTreeDistancesSum = sum([distance(nut, tree) for nut in nuts]) * 2
        totalDistance = totalDistance + nutToTreeDistancesSum

        return totalDistance


def distance(pos1: List[int], pos2: List[int]) -> int:
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])


if __name__ == "__main__":
    """
    Example 1:
    Input:
    Height : 5
    Width : 7
    Tree position : [2,2]
    Squirrel : [4,4]
    Nuts : [[3,0], [2,5]]
    Output: 12
    Explanation:
        Squirrel ([4,4]) -> Nut ([2,5]) -> Tree ([2,2]) -> Nut ([3,0]) -> Tree ([2,2])
                         3       +       3      +        3      +      3                =   12    
    """
    height = 5
    width = 7
    tree = [2, 2]
    squirrel = [4, 4]
    nuts = [[3, 0], [2, 5]]
    print(f"Minimum distance for the squirrel at {squirrel} to collect all the nuts at {nuts} and put them under the "
          f"tree at {tree} one by one: {SquirrelSimulation.minDistance(height, width, tree, squirrel, nuts)}")
