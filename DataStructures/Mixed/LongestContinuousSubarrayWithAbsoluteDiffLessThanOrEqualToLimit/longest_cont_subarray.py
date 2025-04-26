'''
Leetcode 1438: Longest Continuous Subarray With Absolute Diff Less Than or Equal to Limit

Given an array of integers nums and an integer limit, return the size of the longest non-empty subarray such that the absolute difference between any two elements of this subarray is less than or equal to limit.

Example 1:
Input: nums = [8,2,4,7], limit = 4
Output: 2 
Explanation: All subarrays are: 
[8] with maximum absolute diff |8-8| = 0 <= 4.
[8,2] with maximum absolute diff |8-2| = 6 > 4. 
[8,2,4] with maximum absolute diff |8-2| = 6 > 4.
[8,2,4,7] with maximum absolute diff |8-2| = 6 > 4.
[2] with maximum absolute diff |2-2| = 0 <= 4.
[2,4] with maximum absolute diff |2-4| = 2 <= 4.
[2,4,7] with maximum absolute diff |2-7| = 5 > 4.
[4] with maximum absolute diff |4-4| = 0 <= 4.
[4,7] with maximum absolute diff |4-7| = 3 <= 4.
[7] with maximum absolute diff |7-7| = 0 <= 4. 
Therefore, the size of the longest subarray is 2.
'''

from typing import List
import heapq

class Solution:
    '''
    - Time Complexity: O(n⋅logn):
      - Initializing the two heaps takes O(1) time.
      - Iterating through the array nums from left to right involves a single loop that runs n times.
      - Adding each element to the heaps takes O(logn) time per operation due to the properties of heaps. Over the entire array, this results in O(n⋅logn) time for both heaps combined.
      - Checking the condition and potentially shrinking the window involves comparing the top elements of the heaps and moving the left pointer. Removing elements from the heaps that are outside the current window also takes O(logn) time per operation. Over the entire array, this results in O(n⋅logn) time.
      - Updating the maxLength variable involves a simple comparison and assignment, each taking O(1) time per iteration. Over the entire array, this takes O(n) time.
      - Therefore, the total time complexity is O(n⋅logn).

    - Space Complexity: O(n):
      - The two heaps, maxHeap and minHeap, store elements of the array along with their indices. In the worst case, each heap could store all n elements of the array.
      - Therefore, the total space complexity is O(n).
    '''
    def longestSubarray(self, nums: List[int], limit: int) -> int:
        # Keep max and min elements in max heap and min heap, O(1) accessible
        max_heap = []
        min_heap = []

        # Left index, maximum length to keep track
        left = 0
        max_length = 0

        # Iterate over the array nums with index right
        for right in range(len(nums)):
            heapq.heappush(max_heap, (-nums[right], right))
            heapq.heappush(min_heap, (nums[right], right))

            # Check if abs diff between max min exceeds the limit
            # First value is negative, because we used negative value for max heap
            while -max_heap[0][0] - min_heap[0][0] > limit:
                # Move left pointer to the right until the condition is satisfied
                # We need the min, because we don't know if the left index has min or max val
                # But we know that left index candidate is the minimum of index of max and min elements
                left = min(max_heap[0][1], min_heap[0][1]) + 1

                # Remove elements from the heaps that are outside current window
                while max_heap[0][1] < left:
                    heapq.heappop(max_heap)
                while min_heap[0][1] < left:
                    heapq.heappop(min_heap)

            # Update max_length with the length of the current valid window
            max_length = max(max_length, right - left + 1)

        return max_length