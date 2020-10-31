/**
 Given an array with n integers, your task is to check if it could become non-decreasing by modifying at most 1 element.
 We define an array is non-decreasing if array[i] <= array[i + 1] holds for every i (1 <= i < n).

 Example 1:
 Input: [4,2,3]
 Output: True
 Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

 Example 2:
 Input: [4,2,1]
 Output: False
 Explanation: You can't get a non-decreasing array by modify at most one element.

 Note: The n belongs to [1, 10,000].
 */

import java.util.*;

public class NonDecreasingArray {
    /**
     * Keep a counter for number of changes to turn the array into non-decreasing array.
     * If counter > 1, return false. Else, true.
     * @param nums
     * @return
     */
    public static boolean nonDecreasingArrayWithAtMostOneChange(int[] nums) {
        int counter = 0;
        if(nums.length < 3){
            return true;
        }
        // Iterate over array elements
        for(int i = 0; i < nums.length-2; i++){
            //nums[i]  nums[i+1]  nums[i+2]
            if(nums[i] <= nums[i+1]){
                if(nums[i+1] > nums[i+2]){
                    if(nums[i] <= nums[i+2]){
                        nums[i+1] = nums[i];
                    } else if(nums[i] > nums[i+2]){
                        nums[i+2] = nums[i+1];
                    }
                    counter++;
                }
            } else if(nums[i] > nums[i+1]){
                if(nums[i+1] > nums[i+2]){
                    //return false;
                    counter = counter + 2;
                } else {
                    if(nums[i] > nums[i+2]){
                        nums[i] = nums[i+1];
                    } else if (nums[i] <= nums[i+2]){
                        nums[i+1] = nums[i];
                    }
                    counter++;
                }
            }
            if(counter > 1){
                return false;
            }
        }
        return (counter > 1) ? false : true;
    }

    /**
     * Run test cases
     * @param nums
     */
    public static void runTestCases(int[] nums) {
        System.out.println("Is it possible to convert " + Arrays.toString(nums) + " to a nondecreasing array by modifying "
                            + "at most one element?: " + (nonDecreasingArrayWithAtMostOneChange(nums) ? "Yes" : "No"));
    }

    public static void main(String[] args){
        // Test 1
        int[] nums = {4,2,3};
        runTestCases(nums);

        // Test 2
        int[] nums2 = {3,4,2,3};
        runTestCases(nums2);
    }
}