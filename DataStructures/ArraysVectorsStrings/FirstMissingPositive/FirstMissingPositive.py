"""
Note: This is Leetcode question 41: https://leetcode.com/problems/first-missing-positive/

Given an unsorted integer array nums, find the smallest missing positive integer.

Example 1:
Input: nums = [1,2,0]
Output: 3

Example 2:
Input: nums = [3,4,-1,1]
Output: 2

Example 3:
Input: nums = [7,8,9,11,12]
Output: 1


Constraints:
1) 0 <= nums.length <= 300
2) -231 <= nums[i] <= 231 - 1


Follow up: Could you implement an algorithm that runs in O(n) time and uses constant extra space?
"""
from typing import List


class FirstMissingPositive:
    @staticmethod
    def firstMissingPositive(nums: List[int]) -> int:
        positiveSet = set()

        # Add positive integers to a set
        for num in nums:
            if num > 0:
                positiveSet.add(num)

        # Iterate over integers 1..len(nums), find first number not in positiveSet
        for num in range(1, len(nums)+1, 1):
            if num not in positiveSet:
                return num

        # If all numbers in the array exist, 1 plus length of array will not exist, return it
        return len(nums)+1

    # TODO: There is also a O(n) time and O(1) space solution here, which is an overkill.
    #  Main idea is to use array elements as index, and mark them as negative if thet are present.
    #  Link 1: https://leetcode.com/problems/first-missing-positive/solution/
    #  Link 2: https://www.geeksforgeeks.org/find-the-smallest-positive-number-missing-from-an-unsorted-array/


if __name__ == "__main__":
    """
    Example 1:
    Input: nums = [1,2,0]
    Output: 3
    """
    nums = [1,2,0]
    print(f"First positive integer missing from array {nums} is: {FirstMissingPositive.firstMissingPositive(nums)}")
