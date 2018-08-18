/**
 Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

 For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

 Note:
 You must do this in-place without making a copy of the array.
 Minimize the total number of operations.
 */

// Feel free to check alternative solutions: https://leetcode.com/problems/move-zeroes/solution/

import java.util.*; // For Arrays.toString() method

public class MoveZeroes {
    /**
     * Keep a pointer "nonZeroPointer" to next index of non-zero value, and another pointer "current" to traverse the array
     * While iterating on the array with "current", if "nums[current]" is non-zero, swap "nums[current]" and "nums[nonZeroPointer]"
     * At the end, all zeroes will be on the right, and all non-zero values will be on the left, with preserved order.
     *
     * Invariant 1: All elements before "nonZeroPointer" pointer are non-zeroes.
     * Invariant 2: All elements between the current and "nonZeroPointer" pointer are zeroes.
     *
     * Time complexity: O(n)
     * Space complexity: O(1)
     *
     * @param nums
     */
    public static void moveZeroes(int[] nums) {
        // Pointer to the next index of non-zero value
        int nonzeroPointer = 0;

        // Traverse the array with "current" pointer.
        for(int current = 0; current < nums.length; current++) {
            // if "nums[current]" is non-zero, swap "nums[current]" and "nums[nonZeroPointer]"
            if(nums[current] != 0) {
                // Swap nums[current] and nums[nonzeroPointer]
                int value = nums[nonzeroPointer];
                nums[nonzeroPointer] = nums[current];
                nums[current] = value;

                // Increment nonzeroPointer
                nonzeroPointer++;
            }
        }
    }

    public static void runTestCase(int[] numbers){
        int[] originalNumbers = Arrays.copyOf(numbers, numbers.length);
        moveZeroes(numbers);
        System.out.println("Move zeroes to the end of array " + Arrays.toString(originalNumbers ) + ": " + Arrays.toString(numbers));
    }

    public static void main(String[] args){
        // Test 1: Mix of 0s and non-0s
        int[] nums = new int[] {1, 2, 0, 5, 6, 0, 9, 8, 0, 23};
        runTestCase(nums);

        // Test 2: 0s at the beginning, non-zeroes at the end
        int[] nums2 = new int[] {0, 0, 0, 1, 2, 3, 4};
        runTestCase(nums2);

        // Test 3: All zeroes
        int[] nums3 = new int[] {0, 0, 0, 0, 0};
        runTestCase(nums3);

        // Test 4: All non-zeroes
        int[] nums4 = new int[] {1, 2, 3, 4, 5};
        runTestCase(nums4);

        // Test 5: Empty array
        int[] nums5 = new int[] {};
        runTestCase(nums5);
    }
}