'''
Leetcode 3434: Maximum Frequency After Subarray Operation

You are given an integer array nums and an integer k. You can perform the following operation on the array:
1. Choose a subarray of length k and increment all elements in that subarray by 1.
2. Repeat this operation any number of times.
Your task is to find the maximum frequency of any element in the array after performing the operation any number of times.
'''

class Solution:
  def maxFrequency(self, nums: list[int], k: int) -> int:
    return nums.count(k) + max(self._kadane(nums, target, k)
                               for target in range(1, 51)
                               if target != k)

  def _kadane(self, nums: list[int], target: int, k: int) -> int:
    """
    Returns the maximum achievable frequency of `k` by Kakane's algorithm,
    where each `target` in subarrays is transformed to `k`.
    """
    maxSum = 0
    sum = 0
    for num in nums:
      if num == target:
        sum += 1
      elif num == k:
        sum -= 1
      if sum < 0:  # Reset sum if it becomes negative (Kadane's spirit).
        sum = 0
      maxSum = max(maxSum, sum)
    return maxSum