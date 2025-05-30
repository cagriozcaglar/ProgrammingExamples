'''
Leetcode 1814: Count Nice Pairs in an Array

You are given an array nums that consists of non-negative integers. Let us define rev(x) as the reverse of the non-negative integer x. For example, rev(123) = 321, and rev(120) = 21. A pair of indices (i, j) is nice if it satisfies all of the following conditions:
* 0 <= i < j < nums.length
* nums[i] + rev(nums[j]) == nums[j] + rev(nums[i])

Return the number of nice pairs of indices. Since that number can be too large, return it modulo 109 + 7.

Example 1:
Input: nums = [42,11,1,97]
Output: 2
Explanation: The two pairs are:
 - (0,3) : 42 + rev(97) = 42 + 79 = 121, 97 + rev(42) = 97 + 24 = 121.
 - (1,2) : 11 + rev(1) = 11 + 1 = 12, 1 + rev(11) = 1 + 11 = 12.
'''
from typing import List
from collections import defaultdict

class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        num_rev_diffs = [num - int(str(num)[::-1]) for num in nums]
        count = 0
        MOD = 10 ** 9 + 7
        value_count_map = defaultdict(int)

        for num in num_rev_diffs:
            count = (count + value_count_map[num]) % MOD
            value_count_map[num] += 1

        return count