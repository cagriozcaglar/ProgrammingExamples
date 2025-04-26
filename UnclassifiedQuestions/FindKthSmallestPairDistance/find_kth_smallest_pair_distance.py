'''
Leetcode 719: Find K-th Smallest Pair Distance

The distance of a pair of integers a and b is defined as the absolute difference between a and b.

Given an integer array nums and an integer k, return the kth smallest distance among all the pairs nums[i] and nums[j] where 0 <= i < j < nums.length.

Example 1:
Input: nums = [1,3,1], k = 1
Output: 0
Explanation: Here are all the pairs:
(1,3) -> 2
(1,1) -> 0
(3,1) -> 2
Then the 1st smallest distance pair is (1,1), and its distance is 0.

Example 2:
Input: nums = [1,1,1], k = 2
Output: 0

Example 3:
Input: nums = [1,6,1], k = 3
Output: 5
'''
from typing import List

'''
- Let n be the number of elements and M be the maximum possible distance.

- Time complexity: O(n*log(M) + n*log(n))
  - Sorting the array takes O(n*log(n)) (2nd term above).
  - The O(nlogM) term arises from the binary search over possible distances, where the search space is up to the maximum possible distance M. For each mid-value in the binary search, the countPairsWithMaxDistance function is called, which takes O(n).
  - The binary search itself runs in O(logM) time.
  - Hence, the combined time complexity is O(n*log(M) + n*log(n)), where the binary search and pair counting operations are combined.

- Space complexity: O(S)
  - The space complexity is constant because the algorithm only uses a fixed amount of extra space for the left and right pointers, the mid-value, and counters. It does not require additional data structures that scale with the input size, so the space complexity is O(1), excluding the space used to store the input array.
  - Some extra space is used when we sort an array of size n in place. The space complexity of the sorting algorithm (S) depends on the programming language. The value of S depends on the programming language and the sorting algorithm being used:
    - In Python, the sort() method sorts a list using the Timsort algorithm which is a combination of Merge Sort and Insertion Sort and has a space complexity of O(n)
    - In C++, the sort() function is implemented as a hybrid of Quick Sort, Heap Sort, and Insertion Sort, with a worst-case space complexity of O(logn)
    - In Java, Arrays.sort() is implemented using a variant of the Quick Sort algorithm which has a space complexity of O(logn)
Thus, the total space complexity of the algorithm is O(S).
'''
class Solution:
    def smallestDistancePair(self, nums: List[int], k: int) -> int:
        # Sort array
        nums.sort()

        # Init binary search range on pair distance: [0, max_element-min_element]
        low = 0
        high = nums[-1] - nums[0]

        while low < high:
            mid = (low + high) // 2
            # Count pairs with distance <= mid
            count = self.count_pairs_with_max_distance(nums, mid)
            # Binary search on pair distance
            if count < k:  # Search right
                low = mid + 1
            else:  # Search left
                high = mid

        return low

    # Count number of pairs with distance <= max_distance using a sliding window
    def count_pairs_with_max_distance(self, nums: List[int], max_distance: int) -> int:
        count = 0
        left = 0

        for right in range(len(nums)):
            # Update left to maintain window with distance <= max_distance
            while nums[right] - nums[left] > max_distance:
                left += 1
            # Add number of valid pairs ending at current value of right pointer. Not
            # adding +1, because left wasn't incremented in last iteration of while above
            count += right - left

        return count