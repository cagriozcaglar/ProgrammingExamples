/**
 * Given an array of 0s and 1s, sort the array.
 *
 * Example solutions: https://www.geeksforgeeks.org/segregate-0s-and-1s-in-an-array-by-traversing-array-once/
 */

import java.util.*; // For Arrays.toString()

public class Sort0sand1s {

    /**
     * Most optimal solution, O(n) time.
     * Keep two pointers, left and right. Do the following while left < right:
     *  1) While numbers[left] == 0 and left < right, increment left.
     *  2) While numbers[right] == 1, and left < right, decrement right.
     *  3) If left < right, swap numbers[left] and numbers[right]
     * Time complexity: O(n)
     * One pass over the array
     * @param numbers
     */
    public static void sort0sand1s(int[] numbers) {
        int left = 0;
        int right = numbers.length - 1;

        while(left < right){
            // Increment left index if the value is 0
            while(numbers[left] == 0 && left < right){
                left++;
            }

            // decrement right index if the value is 1
            while(numbers[right] == 1 && left < right){
                right--;
            }

            // If left < right, then there is a 1 at the left and 0 on the right.
            // Swap numbers[left] and numbers[right]
            if(left < right){
                numbers[left] = 0;
                numbers[right] = 1;
                left++;
                right--;
            }
        }
    }

    /**
     * Solution with two passes over the array.
     * 1) In the first pass, get the number of 0s in the array.
     * 2) In the second pass, Fill the first zeroCount indices of the array with 0s, and the remaining oneCount indices of
     * the array with 1s.
     *
     * Time Complexity: O(n)
     * Two passes over the array
     * @param numbers
     */
    public static void sort0sand1sWithTwoPasses(int[] numbers) {
        // 1. Get the number of 0s in the array: Requires one pass on the array
        int zeroCount = 0;
        for(int number : numbers){
            if(number == 0) {
                zeroCount++;
            }
        }
        // 2. Second pass: Fill the first zeroCount indices of the array with 0s, and the remaining oneCount indices of the array with 1s
        // 2.1. Fill the first zeroCount indices of the array with 0s
        int index = 0;
        for(; index < zeroCount; index++) {
            numbers[index] = 0;
        }
        // 2.2. Fill the remaining oneCount indices of the array with 1s
        for(; index < numbers.length; index++) {
            numbers[index] = 1;
        }
    }

    public static void runTestCase(int[] numbers) {
        System.out.println("Original array: " + Arrays.toString(numbers)); // Arrays.toString() requires importing java.util.*
        // Create array copy for the call to the second function
        int[] numbers2 = new int[numbers.length];
        System.arraycopy(numbers, 0, numbers2, 0, numbers.length);
        // Solution 1: One-pass solution
        sort0sand1s(numbers);
        System.out.println("Sorted array with one-pass solution: " + Arrays.toString(numbers));
        // Solution 2: two-pass solution
        sort0sand1sWithTwoPasses(numbers);
        System.out.println("Sorted array with one-pass solution: " + Arrays.toString(numbers));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        int[] numbers = new int[]{0, 1, 1, 0, 0, 0, 1, 1, 0};
        runTestCase(numbers);

        // Test 2
        int[] numbers2 = new int[]{0, 1, 0, 1, 0, 1, 1};
        runTestCase(numbers2);

        // Test 3
        int[] numbers3 = new int[]{1, 1, 1, 1, 0, 0, 0};
        runTestCase(numbers3);
    }
}