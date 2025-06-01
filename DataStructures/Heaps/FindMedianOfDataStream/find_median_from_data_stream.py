'''
Leetcode 295: Find Median from Data Stream

The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.
For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.

Example 1:
Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]
Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0
'''

# A max-heap to store the smaller half of the input numbers
# A min-heap to store the larger half of the input numbers
# If the following conditions are met:
# 1) Both the heaps are balanced (or nearly balanced)
# 2) The max-heap contains all the smaller numbers while the min-heap contains all the larger numbers

"""
Adding a number num:
1) Add num to max-heap lo. Since lo received a new element, we must do a balancing step for hi. So remove the largest element from lo and offer it to hi.
2) The min-heap hi might end holding more elements than the max-heap lo, after the previous operation. We fix that by removing the smallest element from hi and offering it to lo.

- Time complexity: O(5⋅logn)+O(1)≈O(logn).
At worst, there are three heap insertions and two heap deletions from the top. Each of these takes about O(logn) time.
Finding the median takes constant O(1) time since the tops of heaps are directly accessible.

- Space complexity: O(n) linear space to hold input in containers.
"""

from typing import heapq

# Example solution: https://leetcode.com/problems/find-median-from-data-stream/discuss/696658/Python-Logic-Explained-with-2-Heaps-Clean-code.
class MedianFinder:

    def __init__(self):
        """
        initialize your data structure here.
        """
        # LowerHalf is max-heap, holding max of lower half.
        # Use negative of keys, because default is min-heap in python
        self.lowerHalf = []
        # higherHalf is min-heap, holding min of lower half.
        self.higherHalf = []

    # NOTE / CAREFUL: When lowerHalf and higherHalf interacts, and when pushing to / popping from lowerHalf, which is a maxHeap, push the negative of num, because by default in Python, heaps are min-heap
    def addNum(self, num: int) -> None:
        # Assumption: size(lowerHalf) - size(higherHalf) is in range [0, 1]. Therefore, we assume lowerHalf needs more elements. So, we push to lowerHalf first, then operate.
        # 1. Push to left half first
        heapq.heappush(self.lowerHalf, -num)
        
        # 2. Balancing: Pop from left, push to right
        heapq.heappush(self.higherHalf, -heapq.heappop(self.lowerHalf))

        # 3. If left size becomes smaller, pop from right, push to left
        if len(self.lowerHalf) < len(self.higherHalf):
            heapq.heappush(self.lowerHalf, -heapq.heappop(self.higherHalf))
 
    def findMedian(self) -> float:
        # NOTE / CAREFUL: When using max value in lowerHalf, use the negative of num, because by default in Python, heaps are min-heap, and you pushed negative of values
        # Odd number
        if len(self.lowerHalf) > len(self.higherHalf):
            return -self.lowerHalf[0]
        # Even number
        else:
            return float(-self.lowerHalf[0] + self.higherHalf[0]) / 2.0

'''
Adding number 41
MaxHeap lo: [41]           // MaxHeap stores the largest value at the top (index 0)
MinHeap hi: []             // MinHeap stores the smallest value at the top (index 0)
Median is 41
=======================
Adding number 35
MaxHeap lo: [35]
MinHeap hi: [41]
Median is 38
=======================
Adding number 62
MaxHeap lo: [41, 35]
MinHeap hi: [62]
Median is 41
=======================
Adding number 4
MaxHeap lo: [35, 4]
MinHeap hi: [41, 62]
Median is 38
=======================
Adding number 97
MaxHeap lo: [41, 35, 4]
MinHeap hi: [62, 97]
Median is 41
=======================
Adding number 108
MaxHeap lo: [41, 35, 4]
MinHeap hi: [62, 97, 108]
Median is 51.5
'''