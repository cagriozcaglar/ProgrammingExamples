'''
Leetcode 528. Random Pick With Weight

You are given a 0-indexed array of positive integers w where w[i] describes the weight of
the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range
[0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is
w[i] / sum(w).

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25
(i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).
'''

# Time complexity: O(n) + O(log(n)) = O(n)
# Space complexity: O(n)
import random
import itertools
from typing import List

class Solution:
    def __init__(self, w: List[int]):
        self.w = w
        self.sumval = sum(self.w)
        self.cumsum = [x / self.sumval for x in list(itertools.accumulate(w))]


    def pickIndex(self) -> int:
        num = random.random()
        left, right = 0, len(self.cumsum) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.cumsum[mid] < num:
                left = mid + 1
            elif self.cumsum[mid] > num:
                right = mid - 1
            else:
                return mid
        return left


# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()