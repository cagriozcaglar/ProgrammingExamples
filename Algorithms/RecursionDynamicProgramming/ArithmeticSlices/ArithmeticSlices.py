"""
Arithmetic Slices
Recursion, DP, formula solution
"""
from typing import List
class Solution:
    """
    Using Formula
    """
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        count = 0
        theSum = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                count += 1
            else:
                theSum += count * (count+1) // 2
                count = 0
        theSum += count * (count+1) // 2
        return theSum

    """
    DP, Constant Space
    """
    def numberOfArithmeticSlicesDpConstantSpace(self, nums: List[int]) -> int:
        dp = 0
        count = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp = 1 + dp
                count += dp
            else:
                dp = 0
        return count

    """
    DP
    """
    def numberOfArithmeticSlicesDp(self, nums: List[int]) -> int:
        """
         * dp[i]: Number of arithmetic slices possible in the range (k,i) and not in any range (k,j) such that j<i.
         * k: minimum index possible such that (k,j) is a valid Arithmetic Slice.
        """
        dp = [0] * len(nums)
        count = 0
        for i in range(2, len(nums)):
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                dp[i] = 1 + dp[i-1]
                count += dp[i]
        return count

    """
    Recursion
    """
    def numberOfArithmeticSlicesNoWork(self, nums: List[int]) -> int:
        def slices(nums: List[int], i: int) -> int:
            nonlocal theSum
            """
            Returns the number of Arithmetic Slices in the range (k,i), but which are not a part of any range (k,j) such that j<i
            """
            if i<2:
                return 0
            ap = 0
            print(f"i: {i}")
            if nums[i] - nums[i-1] == nums[i-1] - nums[i-2]:
                ap = 1 + slices(nums, i-1)
                theSum += ap
            else:
                slices(nums, i-1)
            return ap

        # Main
        theSum = 0
        slices(nums, len(nums)-1)
        return theSum