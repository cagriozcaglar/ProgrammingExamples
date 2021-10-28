"""
Note: This question is in Leetcode: https://leetcode.com/problems/trapping-rain-water/

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it
can trap after raining.

Example 1:
Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case,
6 units of rain water (blue section) are being trapped.

Example 2:
Input: height = [4,2,0,3,2,5]
Output: 9

Constraints:
 - n == height.length
 - 0 <= n <= 3 * 104
 - 0 <= height[i] <= 105

Note: Best video solution on this one: https://www.youtube.com/watch?v=RV7jsfvJ33U&ab_channel=Knapsak
"""
from typing import List
from collections import deque


class TrappingRainWater:
    @staticmethod
    def trapDynamicProgramming(height: List[int]) -> int:
        """
        Dynamic Programming Solution
         - Time complexity: O(n)
         - Space complexity: O(n)
        :param height:
        :return:
        """
        # Error check
        if height is None or len(height) == 0:
            return 0

        # Initialize leftMaxSoFar and rightMaxSoFar arrays
        leftMaxSoFar, rightMaxSoFar = [0] * len(height), [0] * len(height)
        # Total water
        totalWater = 0

        # Populate leftMaxSoFar
        leftMaxSoFar[0] = height[0]
        for i in range(1, len(leftMaxSoFar)):
            leftMaxSoFar[i] = max(height[i], leftMaxSoFar[i-1])

        # Populate rightMaxSoFar
        rightMaxSoFar[len(height)-1] = height[len(height)-1]
        # Start index: len(rightMaxSoFar)-2 (included), Last index: -1(excluded), Step size: -1
        for i in range(len(rightMaxSoFar)-2, -1, -1):
            rightMaxSoFar[i] = max(height[i], rightMaxSoFar[i+1])

        # Calculate trapped water at each index
        for i in range(0, len(height)):
            totalWater += min(leftMaxSoFar[i], rightMaxSoFar[i]) - height[i]

        return totalWater

    @staticmethod
    def trapWithStack(height: List[int]) -> int:
        """
        Solution with Stack
         - Time complexity: O(n)
         - Space complexity: O(n)
        :param height:
        :return:
        """
        # Error check
        if height is None or len(height) == 0:
            return 0

        # Hold indices of height array
        indexTrackerStack = deque()
        # totalWater to be returned as the answer
        totalWater = 0

        for i in range(0, len(height)):
            while len(indexTrackerStack) != 0 and height[i] > height[indexTrackerStack[-1]]:
                topIndex = indexTrackerStack[-1]
                indexTrackerStack.pop()
                if len(indexTrackerStack) == 0:
                    break
                distance = i - indexTrackerStack[-1] - 1
                bounded_height = min(height[i], height[indexTrackerStack[-1]]) - height[topIndex]
                totalWater += distance * bounded_height
            indexTrackerStack.append(i)
        return totalWater

    # TODO: Implement optimal solution with two pointers: Time complexity: O(n), Space complexity: O(1))
    def trapWithTwoPointers(self, height: List[int]) -> int:
        """
        Solution with Two pointers
         - Time complexity: O(n)
         - Space complexity: O(1)
        :param height:
        :return:
        """
        pass


if __name__ == "__main__":
    """
    Example 1:
    Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
    Output: 6
    """
    height = [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]
    print(f"Total trapped rain water with dynamic programming solution for {height}: "
          f"{TrappingRainWater.trapDynamicProgramming(height)}")
    print(f"Total trapped rain water with stack solution for {height}: "
          f"{TrappingRainWater.trapWithStack(height)}")
    print("")

    """
    Example 2:
    Input: height = [4,2,0,3,2,5]
    Output: 9
    """
    height = [4, 2, 0, 3, 2, 5]
    print(f"Total trapped rain water with dynamic programming solution for {height}: "
          f"{TrappingRainWater.trapDynamicProgramming(height)}")
    print(f"Total trapped rain water with stack solution for {height}: "
          f"{TrappingRainWater.trapWithStack(height)}")
    print("")