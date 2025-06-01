'''
Leetcode 347: Top K Frequent Elements

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
'''
from typing import List
from collections import Counter
import heapq

class Solution:
    # Solution 3: Max-Heap
    # Let's start from the simple heap approach with O(Nlogk) time complexity.
    # To ensure that O(Nlogk) is always less than O(NlogN),
    # the particular case k=N could be considered separately and solved in O(1) time.
    # Time: O(n.log(k))
    # Space: O(N + k)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]: 
        # O(1) time 
        if k == len(nums):
            return nums
        
        # 1. Build hash map: character and how often it appears
        # O(N) time
        count = Counter(nums)   
        # 2-3. Build heap of top k frequent elements and
        # convert it into an output array
        # O(N log k) time
        return heapq.nlargest(k, count.keys(), key=count.get)

    # Solution 2: Min-Heap
    # Min-heap, because I need top-k, and condition has to use smallest of these top-k frequent
    # Top element of min-heap is the *least* frequnt of min-heap of size k
    # Time: O(nlog(k))
    # Space: O(N+k)
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        num_freqs = [[freq, num] for num, freq in dict(Counter(nums)).items()]
        min_heap = num_freqs[:k]
        # Insert first k elements only, to min-heap
        heapq.heapify(min_heap)

        # For the remaining n-k elements: if element has higher frequency than top,
        # pop top element, push new element to heap
        for freq, num in num_freqs[k:]:
            # If item is more freq, push new, pop top, size remains the same, k.
            if min_heap[0][0] < freq:
                heapq.heappushpop(min_heap, [freq, num])

        # The following works, because we kept the size of heap as k
        return [num for freq, num in min_heap]
        ### NOTE: The following works, too. But it is unnecessary extra work.
        ### Because it calls heapify again, and then outputs top-k frequent.
        # return [num for [freq, num] in heapq.nsmallest(k, min_heap)]

    # Solution 1: Sort by decreasing frequency, take first k
    # Time: O(n log(n))
    # Space: Depends on sorting algorithm. E.g. heap-sort in-place, quicksort in-place
    def topKFrequentSorting(self, nums: List[int], k: int) -> List[int]:
        num_freqs = [[num, freq] for num, freq in dict(Counter(nums)).items()]
        num_freqs.sort(key=lambda x: x[1], reverse=True)

        return [num for num, freq in num_freqs[:k]]