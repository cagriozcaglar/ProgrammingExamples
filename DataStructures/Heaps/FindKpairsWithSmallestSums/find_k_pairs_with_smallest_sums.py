'''
Min-heap solution (two-pointer solution doesn't work)

Here, m is the size of nums1 and n is the size of nums2.
- Time complexity: O(min(k⋅logk,m⋅n⋅log(m⋅n)))
  - We iterate O(min(k,m⋅n)) times to get the required number of pairs.
  - The visited set and heap both can grow up to a size of O(min(k,m⋅n)) because at each iteration we are inserting at most two pairs and popping one pair. Insertions into a min-heap take an additional log factor. So, to insert O(min(k,m⋅n)) elements into minHeap, we need O(min(k⋅logk,m⋅n⋅log(m⋅n)) time.
  - The visited set takes on an average constant time and hence will take O(min(k,m⋅n)) time in major languages like Java and Python except in C++ where it would also take O(min(k⋅logk,m⋅n⋅log(m⋅n))) because we used ordered_set that keeps the values in sorted order.

- Space complexity: O(min(k,m⋅n))
  - The visited set and heap can both grow up to a size of O(min(k,m⋅n)) because at each iteration we are inserting at most two pairs and popping one pair.
'''
from typing import List
from heapq import heappush, heappop
import heapq

class Solution:
    def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
        m, n = len(nums1), len(nums2)

        answer = []
        visited = set()

        min_heap = [(nums1[0] + nums2[0], (0, 0))]
        visited.add((0, 0))

        while k > 0 and min_heap:
            val, (i, j) = heapq.heappop(min_heap)
            answer.append([nums1[i], nums2[j]])

            if i+1 < m and (i+1, j) not in visited:
                heapq.heappush(min_heap, (nums1[i+1] + nums2[j], (i+1, j)))
                visited.add((i+1, j))

            if j+1 < n and (i, j+1) not in visited:
                heapq.heappush(min_heap, (nums1[i] + nums2[j+1], (i, j+1)))
                visited.add((i, j+1))

            k -= 1

        return answer