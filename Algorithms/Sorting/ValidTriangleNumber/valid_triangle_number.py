'''
Leetcode 611. Valid Triangle Number

Given an integer array nums, return the number of triplets chosen from the array that can make
triangles if we take them as side lengths of a triangle.

Example 1:
Input: nums = [2,2,3,4]
Output: 3
Explanation: Valid combinations are: 
2,3,4 (using the first 2)
2,3,4 (using the second 2)
2,2,3

Example 2:
Input: nums = [4,2,3,4]
Output: 4

'''
from typing import List

# Time complexity : O(n^2)
# Loop of k and j will be executed O(n^2) times in total, because, we do not reinitialize the
# value of k for a new value of j chosen (for the same i). Thus the complexity will be O(nlogn+n^2) = O(n^2)
# Space complexity : O(logn). Sorting takes O(logn) space.

class Solution:
    def triangleNumber(self, nums: List[int]) -> int:
        nums.sort()
        count = 0

        # (i, j, k) triplets
        for i in range(len(nums) - 2):  # [0, len(nums)-3]
            k = i + 2
            if nums[i] != 0:  # No triangle edge of length 0 allowed.
                for j in range(i + 1, len(nums) - 1):  # [i+1, len(nums)-2]
                    # We can find this right limit by simply traversing the index k's values starting from the
                    # index k=j+1 for a pair (i,j) chosen and stopping at the first value of k not satisfying
                    # the above inequality. Again, the count of elements nums[k] satisfying nums[i]+nums[j]>nums[k]
                    # for the pair of indices (i,j) chosen is given by k−j−1.
                    while k < len(nums) and nums[i] + nums[j] > nums[k]:
                        k += 1
                    count += k - j - 1

        return count