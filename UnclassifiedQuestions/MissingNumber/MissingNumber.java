/**
 Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

 For example,
 Given nums = [0, 1, 3] return 2.

 Note:
 Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?
 */

import java.util.stream.*; // Needed for IntStream for array sum calculation

class MissingNumber {

    /**
     * Solution with bit manipulation.
     * The basic idea is to use XOR operation. Given that a^b^b =a, two xor operations with the same number will eliminate
     * the number and reveal the original number. In this solution, XOR operation is applied to both the index and value
     * of the array. In a complete array with no missing numbers, the index and value should be perfectly corresponding
     * ( nums[index] = index), so in a missing array, what is left finally is the missing number.
     * @param nums
     * @return
     */
    public static int missingNumberWithBitManipulationOptimal(int[] nums) {
        int result = 0;
        for(int i = 0; i < nums.length; i++){
            result ^= i;        // XOR the index
            result ^= nums[i];  // XOR the number at the index
        }
        result ^= nums.length;  // XOR the last index n
        return result;
    }

    /**
     * Solution with simple math, too simple to be meant to be asked in an interview (e.g. Udacity)
     * 1) Calculate the sum of the array
     * 2) Calculate the sum of the numbers from 0..n => n * (n+1) / 2
     * 3) The difference between the sums ( 2)-1) ) is the missing number.
     * Once again, this is a too-simple solution, and an interviewer is most probably not asking this answer.
     * @param nums
     * @return
     */
    public static int missingNumberWithSum(int[] nums) {
        int n = nums.length;
        // Sum array values
        // Check this out: Simple way to sum array values: IntStream.of(nums).sum()
        int actualSum = IntStream.of(nums).sum();
        // Expected sum of numbers from 0 to n
        int expectedSum = n * (n+1) / 2;
        // Difference is the missing number
        return expectedSum - actualSum;
    }

    /**
     * Run all solutions on a given test case
     * @param nums
     */
    public static void runTestCase(int[] nums){
        // Variables for function profiling
        long startTime;
        long endTime;

        // Solution 1: Bit manipulation solution
        startTime = System.nanoTime();
        int missingNumber = missingNumberWithBitManipulationOptimal(nums);
        endTime =  System.nanoTime();
        System.out.println("Missing number with Bit manipulation solution: " + missingNumber + ". It took " +
                           (endTime-startTime) + " nanoseconds.");
        // Solution 2: Mathematical solution
        startTime = System.nanoTime();
        missingNumber = missingNumberWithSum(nums);
        endTime =  System.nanoTime();
        System.out.println("Missing number with Bit manipulation solution: " + missingNumber + ". It took " +
                           (endTime-startTime) + " nanoseconds.");
    }

    public static void main(String[] args){
        // Test 1
        int[] nums = new int[]{0, 1, 3};
        runTestCase(nums);

        // Test 2
        int[] nums2 = new int[]{7, 6, 4, 3, 2, 1, 0};
        runTestCase(nums2);
    }
}
