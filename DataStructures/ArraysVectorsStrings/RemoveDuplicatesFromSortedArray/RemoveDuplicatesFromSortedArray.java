/**
 Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.
 Do not allocate extra space for another array, you must do this in place with constant memory.

 For example,
 Given input array nums = [1,1,2],
 Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't
 matter what you leave beyond the new length.
*/

// Used for Arrays.toString()
import java.util.*;

public class RemoveDuplicatesFromSortedArray {
    /**
     * Solution 1: Keep a pointer (newArrayPointer) for the last index of the deduped array. If two consecutive elements
     * in the array, nums[runner] and nums[runner-1], are different, nums[runner] is a new element, assign nums[runner]
     * to nums[newArrayPointer], then increment newArrayPointer. Return newArrayPointer, as it is one position past the
     * last index of the deduped array.
     *
     * Time complexity: O(n)
     * Space complexity: O(1)
     *
     * @param nums
     * @return
     */
    public static int removeDuplicates1(int[] nums) {
        if(nums.length < 2) { return nums.length; }
        int newArrayPointer = 1; // Index starts from 1 if there is at least 1 element in the array
        for(int runner = 1; runner < nums.length; runner++){
            if(nums[runner] != nums[runner-1]){
                // First set nums[newArrayPointer] to nums[runner], then increment newArrayPointer
                nums[newArrayPointer++] = nums[runner];
            }
        }
        return newArrayPointer; // Not +1, because we are past the lastIndex by 1, and length is equal to lastIndex+1.
    }

    /**
     * Solution 2: Keep a pointer (newArrayPointer) for the last index of the deduped array. Only compare nums[i] and
     * nums[newArrayPointer] at each time, where i runs through all elements of the array. If two values are different,
     * we have a new element in nums[i], assign it to nums[newArrayPointer], increment newArrayPointer. Finally, return
     * newArrayPointer+1.
     *
     * Time complexity: O(n)
     * Space complexity: O(1)
     *
     * @param nums
     * @return
     */
    public static int removeDuplicates2(int[] nums) {
        if(nums.length == 0) { return 0; }
        int newArrayPointer = 0;
        for(int i = 0; i < nums.length; i++){
            if(nums[i] != nums[newArrayPointer]) {
                nums[++newArrayPointer] = nums[i];
            }
        }
        return newArrayPointer+1;
    }

    /**
     * Run a test case with all solutions
     * @param numbers
     */
    public static void runTestCase(int[] numbers){
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
        System.out.println("Length of Deduped Array: " + removeDuplicates1(numbers));
        System.out.println("Final Array: " + Arrays.toString(numbers));

        // Solution 2
        System.out.println("Original Array: " + Arrays.toString(numbers2));
        System.out.println("Length of Deduped Array: " + removeDuplicates2(numbers2));
        System.out.println("Final Array: " + Arrays.toString(numbers2));

        System.out.println();
    }

    public static void main(String[] args){
        // Test case 1
        int[] numbers = new int[] {1,1,2};
        runTestCase(numbers);

        // Test case 2
        int[] numbers2 = new int[] {1,2,3,4,5,5,5,5,6};
        runTestCase(numbers2);

        // Test case 3: All same numbers
        int[] numbers3 = new int[] {1,1,1,1,1,1};
        runTestCase(numbers3);

        // Test case 4: All distinct numbers
        int[] numbers4 = new int[] {1,2,3,4,5,6};
        runTestCase(numbers4);

        // Test case 5: Empty array
        int[] numbers5 = new int[] {};
        runTestCase(numbers5);

        // Test case 6: Array of length 1
        int[] numbers6 = new int[] {1};
        runTestCase(numbers6);
    }
}