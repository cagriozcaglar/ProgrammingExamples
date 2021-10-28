"""
1150. Check If a Number Is Majority Element in a Sorted Array
Given an integer array nums sorted in non-decreasing order and an integer target, return true if target is a majority
element, or false otherwise.
A majority element in an array nums is an element that appears more than nums.length / 2 times in the array.
"""
class Solution:
    def isMajorityElement(self, nums: List[int], target: int) -> bool:

        def getFirstIndex() -> int:
            low = 0
            high = len(nums)-1
            while low < high:
                mid = low + (high-low) // 2
                if nums[mid] == target:
                    high = mid
                else:
                    low = mid+1
            return high if nums[high] == target else -1

        lowerIndex = getFirstIndex()
        print(f"lowerIndex: {lowerIndex}")
        upperIndex = lowerIndex + len(nums)//2 # Odd or even does not matter
        print(f"upperIndex: {upperIndex}")
        return upperIndex < len(nums) and nums[upperIndex] == target