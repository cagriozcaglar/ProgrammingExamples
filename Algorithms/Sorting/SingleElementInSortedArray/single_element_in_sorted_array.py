'''
Leetcode 540: Single Element in a Sorted Array

You are given a sorted array consisting of only integers where every element appears exactly twice, except for one element which appears exactly once.
Return the single element that appears only once.
Your solution must run in O(log n) time and O(1) space.

Example 1:
Input: nums = [1,1,2,3,3,4,4,8,8]
Output: 2

Example 2:
Input: nums = [3,3,7,7,10,11,11]
Output: 10
'''

'''
Approach: Binary Search on Evens Indexes Only
 
It turns out that we only need to binary search on the even indexes. This approach is more elegant than the last, although both are good solutions.
The single element is at the first even index not followed by its pair. We used this property in the linear search algorithm, where we iterated over all of the even indexes until we encountered the first one not followed by its pair.
Instead of linear searching for this index though, we can binary search for it.
After the single element, the pattern changes to being odd indexes followed by their pair. This means that the single element (an even index) and all elements after it are even indexes not followed by their pair. Therefore, given any even index in the array, we can easily determine whether the single element is to the left or to the right.

Algorithm
We need to set up the binary search variables and loop so that we are only considering even indexes. The last index of an odd-lengthed array is always even, so we can set lo and hi to be the start and end of the array.
We need to make sure our mid index is even. We can do this by dividing lo and hi in the usual way, but then decrementing it by 1 if it is odd. This also ensures that if we have an even number of even indexes to search, that we are getting the lower middle (incrementing by 1 here would not work, it'd lead to an infinite loop as the search space would not be reduced in some cases).
Then we check whether or not the mid index is the same as the one after it.

1) If it is, then we know that mid is not the single element, and that the single element must be at an even index after mid. Therefore, we set lo to be mid + 2. It is +2 rather than the usual +1 because we want it to point at an even index.
2) If it is not, then we know that the single element is either at mid, or at some index before mid. Therefore, we set hi to be mid.
Once lo == hi, the search space is down to 1 element, and this must be the single element, so we return it.

 - Time complexity: O(log n)
 - Space complexity: O(1)
'''
class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        low, high = 0, len(nums) - 1

        while low < high:
            mid = low + (high - low) // 2
            # If mid is odd, go to even index in prev index, mid-1
            if mid % 2 == 1:
                mid -= 1
            # mid has its pair to the right
            if nums[mid] == nums[mid + 1]:
                low = mid + 2
            # If mid's right doesn't have the same value, search left, set high = mid
            else:
                high = mid

        return nums[low]