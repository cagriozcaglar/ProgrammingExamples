/**
 Given an unsorted array of integers, find the length of longest continuous increasing subsequence.

 Note: This question asks the length of longest continuous increasing subsequence, not longest increasing subsequence (LCS),
 in which case we have to use recursion / dynamic programming.

 Example 1:
 Input: [1,3,5,4,7]
 Output: 3
 Explanation: The longest continuous increasing subsequence is [1,3,5], its length is 3.
 Even though [1,3,5,7] is also an increasing subsequence, it's not a continuous one where 5 and 7 are separated by 4.

 Example 2:
 Input: [2,2,2,2,2]
 Output: 1
 Explanation: The longest continuous increasing subsequence is [2], its length is 1.
 Note: Length of the array will not exceed 10,000.
 */

import java.util.Arrays;

public class LongestContinuousIncreasingSubsequence {
    /**
     * Find the length of longest continuous increasing subsequence.
     * This solution uses a simple greedy iterative algorithm: Iterate over the array, keeping current length of
     * increasing continuous subsequence (CIS) and maximum length of increasing continuous subsequence (CIS). Once
     * the end of the array is reached, return the maximum length of CIS.
     * @param nums
     * @return length of longest continuous increasing subsequence
     */
    public static int findMaximumLengthOfCIS(int[] nums) {
        // If nums is an empty / null array, return 0.
        // WARNING: Do not forget this check. Without this check, the function returns 1 for empty/null arrays.
        if(nums == null || nums.length == 0){
            return 0;
        }
        // Initial current length and maximum length is 1.
        int maxLength = 1;
        int currentLength = 1;
        // Iterate over the array and update the max and current length of contigous / continuous increasing subsequence
        for(int i = 0; i < nums.length-1; i++) {
            // Increasing
            if(nums[i] < nums[i+1]){
                currentLength++;
            } else { // Non-increasing
                currentLength = 1;
            }

            // Update maximum length: Short version of the code segment commented-out below
            maxLength = Math.max(currentLength, maxLength);
            /*
            if(currentLength > maxLength){
                maxLength = currentLength;
            }
            */
        }
        return maxLength;
    }

    public static void main(String[] args){
        // Test 1: Normal case
        // Syntax hint: Do not forget to declare the array with "new int[]" on the RHS before writing the integers in brackets.
        int[] nums = new int[]{1,3,5,4,7};
        // Syntax hint: For pretty printing of arrays, use Arrays.toString(arrayVariableName) (import java.util.Arrays for this to work)
        System.out.println(Arrays.toString(nums) + ": " + findMaximumLengthOfCIS(nums));

        // Test 2: Edge case: Array with all same values
        int[] nums2 = new int[]{2,2,2,2,2};
        System.out.println(Arrays.toString(nums2) + ": " + findMaximumLengthOfCIS(nums2));

        // Test 3: Edge case: Empty array
        int[] nums3 = new int[]{};
        System.out.println(Arrays.toString(nums3) + ": " + findMaximumLengthOfCIS(nums3));

        // Test 4: Edge case: Null array
        int[] nums4 = null;
        System.out.println(Arrays.toString(nums4) + ": " + findMaximumLengthOfCIS(nums4));
    }
}