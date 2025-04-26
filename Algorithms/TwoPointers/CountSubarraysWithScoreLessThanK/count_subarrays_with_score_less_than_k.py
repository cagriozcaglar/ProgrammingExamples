'''
Leetcode 2302: Count Subarrays With Score Less Than K

The score of an array is defined as the product of its sum and its length.

For example, the score of [1, 2, 3, 4, 5] is (1 + 2 + 3 + 4 + 5) * 5 = 75.
Given a positive integer array nums and an integer k, return the number of non-empty subarrays of nums
whose score is strictly less than k.

Example 1:
Input: nums = [2,1,4,3,5], k = 10
Output: 6
Explanation:
The 6 subarrays having scores less than 10 are:
- [2] with score 2 * 1 = 2.
- [1] with score 1 * 1 = 1.
- [4] with score 4 * 1 = 4.
- [3] with score 3 * 1 = 3. 
- [5] with score 5 * 1 = 5.
- [2,1] with score (2 + 1) * 2 = 6.
Note that subarrays such as [1,4] and [4,3,5] are not considered because their scores are 10 and 36 respectively, while we need scores strictly less than 10.

Example 2:
Input: nums = [1,1,1], k = 5
Output: 5
Explanation:
Every subarray except [1,1,1] has a score less than 5.
[1,1,1] has a score (1 + 1 + 1) * 3 = 9, which is greater than 5.
Thus, there are 5 subarrays having scores less than 5.
'''

from typing import List
from itertools import accumulate

class Solution:
    '''
    Solution 2: Two pointers
    We can use two pointers to maintain a sliding window, so that the sum of the elements
    in the window is less than k. The number of subarrays with the current element as the
    last element is the length of the window, and we add all window lengths to get the answer.

    The time complexity is O(n), where n is the length of the array nums. The space complexity is O(1).

    https://algo.monster/liteproblems/2302
    '''
    def countSubarrays(self, nums: List[int], k: int) -> int:
        # Init variables
        count = 0
        current_sum = 0

        # Pointer 1: start pointer at 0
        start_index = 0

        # Pointer 2: end pointer.
        # Iterate on the array with second pointer
        for end_index, num in enumerate(nums):
            current_sum += num

            # While score >= k, increment start_index to shrink the window,
            # until score becomes less than k. Update current_sum and start_index
            while current_sum * (end_index - start_index + 1) >= k:
                current_sum -= nums[start_index]
                start_index += 1

            # Number of subarrays ending with current number is the length of the current window
            count += end_index - start_index + 1

        return count

    '''
    Solution 1: Prefix Sum + Binary Search

    1) First, we calculate the prefix sum array s of the array nums, where,
    s[i] represents the sum of the first i elements of the array nums.

    2) Next, we enumerate each element of the array nums as the last element of the subarray.
    For each element, we can find the maximum length l such that s[i] - s[i-l]xl < k by binary search.
    The number of subarrays with this element as the last element is l, and we add all l to get the answer.

    The time complexity is O(nlog(n)), and the space complexity is O(n). Here, n is the length of the array.

    https://leetcode.ca/2022-03-20-2302-Count-Subarrays-With-Score-Less-Than-K/
    '''
    def countSubarraysPrefixSumBinarySearch(self, nums: List[int], k: int) -> int:
        '''
        1. Generate prefix_sum[0..n-1] array from nums[0..n-1] array
        2. Calculate sums of substring(i, j) as prefix_sum[j] - prefix_sum[i-1]
        3. suffix
        '''
        # initial = 0 adds 0 to the beginning of the prefix sum list, e,g. [1,2,3,4] => [0, 1, 3, 6, 10]
        prefix_sums = list(accumulate(nums, initial=0))
        count = 0

        for i in range(1, len(prefix_sums)):
            left, right = 0, i
            while left < right:
                mid = (left + right + 1) // 2
                if (prefix_sums[i] - prefix_sums[i - mid]) * mid < k:
                    left = mid
                else:
                    right = mid - 1
            count += left

        return count
