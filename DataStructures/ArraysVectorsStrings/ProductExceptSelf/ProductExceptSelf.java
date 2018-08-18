/**
 Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of
 all the elements of nums except nums[i].

 Solve it ***** without division ***** and in O(n).

 For example, given [1,2,3,4], return [24,12,8,6].

 Follow up:
 Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose
 of space complexity analysis.)
 */

import java.util.*;

public class ProductExceptSelf {
    /**
     * Method 1: In order to do this without division, given original array A[0..n-1], we create a new array B[0..n-1]
     * as follows: B[j] = A[0] * A[1] * ... * A[j-1]    *    A[j+1] * ... * A[n-1]. The first half of RHS is the set of
     * mult(A[0...j-1]), and second half of RHS is the set of mult(A[j+1...n-1]). So, we create two arrays: one for
     * multiplication of all values on the left, and one for multiplication of all values on the right. Then, we multiply
     * both arrays element-wise, to get the final product values except self.
     *
     * Time complexity: O(n), Space complexity: O(n).
     *
     * @param nums
     * @return
     */
    public static int[] productExceptSelf(int[] nums) {
        int[] leftSideMultiplication = new int[nums.length];
        int[] rightSideMultiplication = new int[nums.length];
        int[] productValuesExceptSelf = new int[nums.length];

        // Fill in left-side-multiplication array
        leftSideMultiplication[0] = 1;
        for(int i = 1; i < nums.length; i++) {
            leftSideMultiplication[i] = nums[i-1] * leftSideMultiplication[i-1];
        }
        // System.out.println("leftSideMultiplication: " + Arrays.toString(leftSideMultiplication));

        // Fill in right-side-multiplication array
        rightSideMultiplication[nums.length-1] = 1;
        for(int i = nums.length-2; i >= 0; i--) {
            rightSideMultiplication[i] = rightSideMultiplication[i+1] * nums[i+1];
        }
        // System.out.println("rightSideMultiplication: " + Arrays.toString(rightSideMultiplication));

        // Generate the final array
        for(int i = 0; i < productValuesExceptSelf.length; i++) {
            productValuesExceptSelf[i] = leftSideMultiplication[i] * rightSideMultiplication[i];
        }

        return productValuesExceptSelf;
    }

    /**
     * Method 2: Same idea as in method 1, but more space-efficient.
     * Example solution: https://leetcode.com/problems/product-of-array-except-self/discuss/65622/Simple-Java-solution-in-O(n)-without-extra-space
     *
     * Time complexity: O(n), Space complexity: O(1).
     *
     * @param nums
     * @return
     */
    public static int[] productExceptSelf2(int[] nums) {
        int[] productValuesExceptSelf = new int[nums.length];

        // Fill in left-side-multiplication array
        productValuesExceptSelf[0] = 1;
        for(int i = 1; i < nums.length; i++) {
            productValuesExceptSelf[i] = nums[i-1] * productValuesExceptSelf[i-1];
        }
        // System.out.println("productValuesExceptSelf: " + Arrays.toString(productValuesExceptSelf));

        // Fill in right-side-multiplication array
        int right = 1;
        for(int i = nums.length-1; i >= 0; i--) {
            productValuesExceptSelf[i] *= right;
            right *= nums[i];
        }
        // System.out.println("productValuesExceptSelf: " + Arrays.toString(productValuesExceptSelf));

        return productValuesExceptSelf;
    }

    /**
     * Run a test case with all methods
     * @param numbers
     */
    public static void runTestCases(int[] numbers) {
        // Method 1
        System.out.println("productExceptSelf(" +  Arrays.toString(numbers) + "): " + Arrays.toString(productExceptSelf(numbers)));
        // Method 2
        System.out.println("productExceptSelf2(" +  Arrays.toString(numbers) + "): " + Arrays.toString(productExceptSelf2(numbers)));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        int[] numbers = new int[] {1,2,3,4,5};
        runTestCases(numbers);

        // Test 2: At least one element in the array is 0
        int[] numbers2 = new int[] {0,1,2,3};
        runTestCases(numbers2);
    }
}