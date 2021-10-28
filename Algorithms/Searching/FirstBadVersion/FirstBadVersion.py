"""
isBadVersion
Easy, Binary Search, modified
"""

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return an integer
# def isBadVersion(version):

def isBadVersion(n: int):
    True

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        low, high = 1, n
        while low < high:   # Not equals
            mid = low + (high - low) // 2  # Prevent overflow
            if isBadVersion(mid):
                high = mid  # not mid-1
            else:
                low = mid+1

        return low