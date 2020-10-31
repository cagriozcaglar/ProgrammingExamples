/**
 Given an array and a value, remove all instances of that value in place and return the new length.
 Do not allocate extra space for another array, you must do this in place with constant memory.
 The order of elements can be changed. It doesn't matter what you leave beyond the new length.

 Example:
 Given input array nums = [3,2,2,3], val = 3

 Your function should return length = 2, with the first two elements of nums being 2.
 */

import java.util.*; // Required for "Arrays.toString()"

public class RemoveElement {
    /**
     * Solution 1: Keep a pointer "newArrayPointer" for the new array. Go through the original array with a "runner"
     * pointer. If nums[runner] is not equal to val, the element should be in the new array, so assign nums[runner] to
     * nums[newArrayPointer], increment newArrayPointer, proceed. Finally, return newArrayPointer, which is equal to
     * last index of new array +1, which is equal to the length of the new array.
     * @param nums
     * @param val
     * @return
     */
    public static int removeElement1(int[] nums, int val) {
        int newArrayPointer = 0;
        for(int runner = 0; runner < nums.length; runner++){
            if(nums[runner] != val){
                nums[newArrayPointer++] = nums[runner];
            }
        }
        return newArrayPointer; // Not +1, because we are already at index which 1 past the last index of the new array
    }

    /**
     * Solution 2: (When elements to remove are rare) Similar to previous solution, but we start the pointer from the end,
     * to account for cases when elements to remove are rare. Consider cases where the array contains few elements to remove.
     * For example, nums = [1,2,3,5,4], val = 4. The previous algorithm will do unnecessary copy operation of the first
     * four elements. Another example is nums = [4,1,2,3,5], val = 4. It seems unnecessary to move elements [1,2,3,5]
     * one step left as the problem description mentions that ** the order of elements could be changed **.
     *
     * Algorithm: Keep new array pointer "newArrayPointer" to point to nums.length (last index + 1). When we encounter
     * nums[runner] = val, we can set nums[runner] to nums[newArrayPointer-1], and dispose the last index by decrementing
     * newArrayPointer. This essentially reduces the final array's size by 1. When runner >= newArrayPointer, the end is
     * reached, and we return newArrayPointer. (Note: runner moves from left to right, newArrayPointer moves from right
     * to left).
     *
     * @param nums
     * @param val
     * @return
     */
    public static int removeElement2(int[] nums, int val) {
        int runner = 0;
        int newArrayPointer = nums.length;
        while(runner < newArrayPointer){
            if(nums[runner] == val) {
                nums[runner] = nums[newArrayPointer-1]; // Because newArrayPointer is initially assigned to array length
                newArrayPointer--; // Decrement final array size
            } else {
                runner++;
            }
        }
        return newArrayPointer;
    }

    public static void runTestCase(int[] numbers, int val){
        // Deep copy the array to a new array first, for other solutions
        int[] numbers2 = new int[numbers.length];
        // NOTE: Check the definition of System.arraycopy(Object src, int srcPos, Object dest, int destPos, int length)
        // - src : This is the source array.
        // - srcPos : This is the starting position in the source array.
        // - dest : This is the destination array.
        // - destPos : This is the starting position in the destination data.
        // - length : This is the number of array elements to be copied.
        // More info: https://www.tutorialspoint.com/java/lang/system_arraycopy.htm
        System.arraycopy(numbers, 0, numbers2, 0, numbers.length);

        // Solution 1
        // NOTE: Arrays.toString() method requires "import java.util.*;"
        System.out.println("Original Array: " + Arrays.toString(numbers));
        int newArrayLength = removeElement1(numbers, val);
        System.out.println("Length of New Array: " + newArrayLength);
        // Create subset array using deep copy.
        int[] numbersSubset = new int[newArrayLength];
        System.arraycopy(numbers, 0, numbersSubset, 0, numbersSubset.length);
        System.out.println("New Array: " + Arrays.toString(numbersSubset));

        // Solution 2
        System.out.println("Original Array: " + Arrays.toString(numbers2));
        int newArrayLength2 = removeElement2(numbers2, val);
        System.out.println("Length of New Array: " + newArrayLength2);
        // Create subset array using deep copy.
        int[] numbersSubset2 = new int[newArrayLength2];
        System.arraycopy(numbers2, 0, numbersSubset2, 0, numbersSubset2.length);
        System.out.println("New Array: " + Arrays.toString(numbersSubset2));

        System.out.println();
    }

    public static void main(String[] args){
        // Test 1
        int[] numbers = new int[] {3,2,2,3};
        int val = 3;
        runTestCase(numbers, val);

        // Test 2: All same elements, which is equal to element to be removed
        int[] numbers2 = new int[] {1,1,1,1,1,1};
        val = 1;
        runTestCase(numbers2, val);

        // Test 3: All distinct elements, which one value which is to be removed
        int[] numbers3 = new int[] {1,2,3,4,5,6};
        val = 4;
        runTestCase(numbers3, val);

        // Test 4: Value not in the array
        int[] numbers4 = new int[] {1,2,3,4,5,6};
        val = 7;
        runTestCase(numbers4, val);

        // Test 5: First values of the array is the element to be removed
        int[] numbers5 = new int[] {3,3,3,3,4,5,6};
        val = 3;
        runTestCase(numbers5, val);
    }
}