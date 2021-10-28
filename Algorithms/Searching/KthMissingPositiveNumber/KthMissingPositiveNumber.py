"""
K-th Positive Missing Number
Binary Search, modified
"""
from typing import List

class Solution:
    def findKthPositive(self, arr: List[int], k: int) -> int:
        low, high = 0, len(arr)-1
        while low <= high:  # CAREFUL: equals sign
            mid = (low+high) // 2
            diff = arr[mid] - mid - 1
            # Right
            if k > diff:
                low = mid+1
            # Left
            elif k <= diff:  # else
                high = mid-1
        # At the end of the loop, left = right + 1,
        # and the kth missing is in-between arr[right] and arr[left].
        # The number of integers missing before arr[right] is
        # arr[right] - right - 1 -->
        # the number to return is
        # arr[right] + k - (arr[right] - right - 1) = k + left
        return arr[high] + (k - (arr[high]-high-1))