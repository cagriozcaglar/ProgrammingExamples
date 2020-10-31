/**
 Given an integer array nums, find the contiguous subarray within an array (containing at least one number) which has the largest product.

 Example 1:
 Input: [2,3,-2,4]
 Output: 6
 Explanation: [2,3] has the largest product 6.

 Example 2:
 Input: [-2,0,-1]
 Output: 0
 Explanation: The result cannot be 2, because [-2,-1] is not a subarray.
 */

import java.util.*;

public class MaximumProductSubarray {
    /**
     * Solution 1: Dynamic Programming, without keeping an array of local minimum and maximum.
     * Great explanation here: https://www.quora.com/How-do-I-solve-maximum-product-subarray-problems.
     * In summary:
     * We know if we multiply two numbers, the product will be large if the numbers are both large in magnitude and are of same sign.
     * 1) Two large positive numbers produce a large positive product
     * 2) Two large negative numbers produce a large positive product
     * 3) A large positive and a large negative number produce a large negative product (which may or may not be used
     *    further with another negative number to produce a large positive product)
     *
     * Thus, to find the maximum product in a subarray, we have to keep track of both local minimum product and local maximum product.
     *
     * In pseudocode, we can do this as follows, mathematically:
     * 1. localMax = max(prevLocalMax ∗ current, current), if current >= 0
     * 2. localMax = max(prevLocalMin ∗ current, current), if current < 0
     * 3. localMin = min(prevLocalMin ∗ current, current), if current >= 0
     * 4. localMin = max(prevLocalMax ∗ current, current), if current < 0
     *
     * Therefore, we only need the last localMin and last localMax, rather than keeping an array of localMin and localMax
     * at each position. Finally, after each iteration, we set globalMax to be the maximum of globalMax and localMax.
     *
     * Time complexity: O(n)
     * Space complexity: O(1) (only localMin, localMax, globalMax variables are used)
     *
     * @param nums
     * @return
     */
    public static int maxProductSubarray1(int[] nums) {
        if(nums.length == 0) {
            return 0;
        }

        // Initialize localMax, localMin, globalMax
        int localMax = nums[0];
        int localMin = nums[0];
        int globalMax = nums[0];

        // Iterate through the rest of the array starting at index 1
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] > 0) {
                // 1. localMax = max(prevLocalMax ∗ current, current), if current >= 0
                localMax = Math.max(localMax * nums[i], nums[i]);
                // 3. localMin = min(prevLocalMin ∗ current, current), if current >= 0
                localMin = Math.min(localMin * nums[i], nums[i]);
            } else {
                // 2. localMax = max(prevLocalMin ∗ current, current), if current < 0
                // CAREFUL: Keep localMax in temp variable, because localMax will be used in localMin calculation next
                int localMaxTemp = Math.max(localMin * nums[i], nums[i]);
                // 4. localMin = max(prevLocalMax ∗ current, current), if current < 0
                localMin = Math.min(localMax * nums[i], nums[i]);
                // CAREFUL: Now, after localMax is used above, copy localMaxTemp to localMax
                localMax = localMaxTemp;
            }
            globalMax = Math.max(globalMax, localMax);
        }
        return globalMax;
    }

    /**
     * Solution 2: Dynamic Programming, this time keeping an array of local minimum and maximum.
     * Algorithm is the same as above. Keeping array of local min and max makes the algorithm more clear, although
     * only the last localMin and localMax values are used in the calculations, as in Solution 1 above.
     *
     * This is similar to the solution here: https://www.programcreek.com/2014/03/leetcode-maximum-product-subarray-java/
     *
     * Time complexity: O(n)
     * Space complexity: O(n) (Arrays for localMin and localMax)
     *
     * @param nums
     * @return
     */
    public static int maxProductSubarray2(int[] nums) {
        if(nums.length == 0) {
            return 0;
        }

        // Initialize localMax, localMin, globalMax
        int[] localMax = new int[nums.length];
        int[] localMin = new int[nums.length];
        localMin[0] = localMax[0] = nums[0];
        int globalMax = nums[0];

        // Iterate through the rest of the array starting at index 1
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] > 0) {
                // 1. localMax = max(prevLocalMax ∗ current, current), if current >= 0
                localMax[i] = Math.max(localMax[i-1] * nums[i], nums[i]);
                // 3. localMin = min(prevLocalMin ∗ current, current), if current >= 0
                localMin[i] = Math.min(localMin[i-1] * nums[i], nums[i]);
            } else {
                // 2. localMax = max(prevLocalMin ∗ current, current), if current < 0
                // NOTE: This time, we are using arrays and we can access the previous local Max by index, so, we
                // don't have to worry about overwriting previous localMax value.
                localMax[i] = Math.max(localMin[i-1] * nums[i], nums[i]);
                // 4. localMin = max(prevLocalMax ∗ current, current), if current < 0
                localMin[i] = Math.min(localMax[i-1] * nums[i], nums[i]);
            }
            globalMax = Math.max(globalMax, localMax[i]);
        }
        return globalMax;
    }

    public static void runTestCase(int[] nums) {
        System.out.println("Maximum product subarray result of array " + Arrays.toString(nums) + " using method 1 is: " + maxProductSubarray1(nums));
        System.out.println("Maximum product subarray result of array " + Arrays.toString(nums) + " using method 2 is: " + maxProductSubarray2(nums));
    }

    public static void main(String[] args) {
        // Test 1
        int[] nums = new int[]{2, 3, -2, 4};
        runTestCase(nums);

        // Test 2
        int[] nums2 = new int[]{7, -2, -4};
        runTestCase(nums2);

        // Test 3: Array with zeros
        int[] nums3 = new int[]{1, 2, 3, 0, 6, 0, -1, -2, -3, 6};
        runTestCase(nums3);

        // Test 4: Array with one positive element
        int[] nums4 = new int[]{1};
        runTestCase(nums4);

        // Test 5: Array with one negative element
        int[] nums5 = new int[]{-5};
        runTestCase(nums5);

        // Test 6: Array with one negative element
        int[] nums6 = new int[]{};
        runTestCase(nums6);
    }
}
