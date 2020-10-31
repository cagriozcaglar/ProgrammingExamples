/**
 Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.
 (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).
 You are given a target value to search. If found in the array return its index, otherwise return -1.
 *** You may assume no duplicate exists in the array. ***
 */

import java.util.*; // For Arrays.toString()

class SearchInRotatedSortedArray {

    /**
     * Solution 1: Iterative solution, one-pass modified binary search.
     * The idea is that when rotating the array, there must be one half of the array that is still in sorted order.
     * For example, 6 7 1 2 3 4 5, the order is disrupted from the point between 7 and 1. So when doing binary search,
     * we can make a judgement that which part is ordered and whether the target is in that range, if yes, continue the
     * search in that half, if not continue in the other half.
     *
     * Time Complexity: O(log(n)) (If no duplicates exist. If duplicates exist, this method runs in O(n) time).
     * @param nums
     * @param target
     * @return
     */
    public static int searchInRotatedSortedArray1(int[] nums, int target) {
        int left = 0;
        int right = nums.length-1;
        while(left <= right){
            int mid = (left + right) / 2;
            // If target is present at middle point, return mid.
            if(nums[mid] == target) {
                return mid;
            }
            // If arr[left..mid] is sorted
            if(nums[left] <= nums[mid]){
                // If target is in [ arr[left], arr[mid] ] range
                if(nums[left] <= target && target < nums[mid]){
                    right = mid - 1; // Search left half
                } else {
                    left = mid + 1;  // Search right half
                }
            } else {
                // If target is in [ arr[mid], arr[high] ] range
                if(nums[mid] < target && target <= nums[right]) {
                    left = mid + 1;   // Search right half
                } else {
                    right = mid-1;    // Search left half
                }
            }
        }
        // If not found, return -1
        return -1;
    }

    /**
     * Solution 2: Recursive solution, one-pass modified binary search
     * Idea is the same as the one in Solution 1 above, except that the binary search in subarrays are done using
     * recursive calls to modified binary search, instead of changing the limits of subarray to search and in an iterative
     * fashion inside a while-loop (as in Solution 1).
     *
     * Time Complexity: O(log(n)) (If no duplicates exist. If duplicates exist, this method runs in O(n) time).
     * @param nums
     * @param target
     * @return
     */
    public static int searchInRotatedSortedArray2(int[] nums, int target) {
        return searchInRotatedSortedArray2Helper(nums, 0, nums.length-1, target);
    }

    /**
     * Helper for Solution 2
     * @param nums
     * @param left
     * @param right
     * @param target
     * @return
     */
    public static int searchInRotatedSortedArray2Helper(int[] nums, int left, int right, int target){
        // Reached the end, target not found, return with -1
        if(left > right) {
            return -1;
        }
        int mid = (left + right) / 2;

        // If target is present at middle point, return mid.
        if(nums[mid] == target) {
            return mid;
        }
        // If arr[left..mid] is sorted
        if(nums[left] <= nums[mid]){
            // If target is in [ arr[left], arr[mid] ] range
            if(nums[left] <= target && target < nums[mid]){
                return searchInRotatedSortedArray2Helper(nums, left, mid-1, target); // Search left half
            } else {
                return searchInRotatedSortedArray2Helper(nums, mid+1, right, target); // Search right half
            }
        } else {
            // If target is in [ arr[mid], arr[high] ] range
            if(nums[mid] < target && target <= nums[right]) {
                return searchInRotatedSortedArray2Helper(nums, mid+1, right, target); // Search right half
            } else {
                return searchInRotatedSortedArray2Helper(nums, left, mid-1, target); // Search left half
            }
        }
    }

    /**
     * Solution 3: Iterative solution, two-pass binary search
     * First binary search is used to find the smallest element, hence the amount of rotation of the array.
     * Second binary search is used to find the target, while accounting for the rotation amount.
     * Each of the two binary search steps here is similar to the iterative method in Solution 1.
     *
     * Time Complexity: O(log(n)) (If no duplicates exist. If duplicates exist, this method runs in O(n) time).
     * @param nums
     * @param target
     * @return
     */
    public static int searchInRotatedSortedArray3(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;

        // 1. Find the index of the smallest value using binary search
        while(left < right){
            int mid = (left + right) / 2;
            // If mid element is larger than right element, order is not preserved
            if(nums[mid] > nums[right]) {
                left = mid+1; // Search right half
            } else { // If mid element <= right element, order is preserved
                right = mid;  // Search left half
            }
        }
        // At this point, left==right is the index of the smallest value, and also the number of places rotated

        // 2. Second binary search, accounting for the rotation using modulo. Similar to Solution 1 above.
        // Set rotation, reset left and right for second binary search
        int rotation = left;
        left = 0;
        right = nums.length - 1;
        while(left <= right){
            int mid = (left + right) / 2;
            int actualMid = (mid + rotation) % nums.length;
            if(nums[actualMid] == target) {
                return actualMid;
            }
            // If mid element is smaller than target, search right by setting left to mid+1
            if(nums[actualMid] < target) {
                left = mid+1;
            } else { // If mid element is <= target, search left by setting right to mid-1
                right = mid-1;
            }
        }
        return -1;
    }

    /**
     * Run a test case with all solutions
     * @param nums
     * @param target
     */
    public static void runTestCase(int[] nums, int target){
        System.out.println("searchInRotatedSortedArray1( " + Arrays.toString(nums) + ", " + target + "): " +
                           searchInRotatedSortedArray1(nums, target));
        System.out.println("searchInRotatedSortedArray2( " + Arrays.toString(nums) + ", " + target + "): " +
                searchInRotatedSortedArray2(nums, target));
        System.out.println("searchInRotatedSortedArray3( " + Arrays.toString(nums) + ", " + target + "): " +
                searchInRotatedSortedArray3(nums, target));

        System.out.println();
    }

    public static void main(String[] args){
        // Test 1
        int[] nums = new int[]{4, 5, 6, 7, 0, 1, 2, 3};
        int target = 2;
        runTestCase(nums, target);

        // Test 2: Target not available in the array
        int[] nums2 = new int[]{6, 7, 8, 9, -1, 0, 1, 2, 4};
        target = 5;
        runTestCase(nums2, target);

        // Test 3: Array with duplicates, searching for non-duplicate target
        int[] nums3 = new int[]{6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5};
        target = 7;
        runTestCase(nums3, target);

        // Test 4: Array with duplicates, searching for duplicate target
        int[] nums4 = new int[]{6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5};
        target = 3;
        runTestCase(nums4, target);
    }
    /**
     Output:
     searchInRotatedSortedArray1( [4, 5, 6, 7, 0, 1, 2, 3], 2): 6
     searchInRotatedSortedArray2( [4, 5, 6, 7, 0, 1, 2, 3], 2): 6
     searchInRotatedSortedArray3( [4, 5, 6, 7, 0, 1, 2, 3], 2): 6

     searchInRotatedSortedArray1( [6, 7, 8, 9, -1, 0, 1, 2, 4], 5): -1
     searchInRotatedSortedArray2( [6, 7, 8, 9, -1, 0, 1, 2, 4], 5): -1
     searchInRotatedSortedArray3( [6, 7, 8, 9, -1, 0, 1, 2, 4], 5): -1

     searchInRotatedSortedArray1( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 7): 1
     searchInRotatedSortedArray2( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 7): 1
     searchInRotatedSortedArray3( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 7): 1

     searchInRotatedSortedArray1( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 3): 11
     searchInRotatedSortedArray2( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 3): 11
     searchInRotatedSortedArray3( [6, 7, 8, 8, 8, 8, 9, 10, 1, 2, 3, 3, 3, 4, 4, 5], 3): 11
     */
}