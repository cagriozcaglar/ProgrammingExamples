/**
 Given an array of integers, every element appears twice except for one. Find that single one.
 Note: Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
 */

import java.util.stream.*;  // In order to use IntStream
import java.util.*;

class SingleNumber {
    /**
     * Solution 1: HashSet
     * Time complexity: O(n)
     * Space complexity: O(n)
     * @param nums
     * @return
     */
    public static int singleNumber1(int[] nums) {
        HashSet<Integer> numbers = new HashSet<Integer>();
        for(int num : nums) {
            // Number appears second time, remove from the set
            if(numbers.contains(num)){
                numbers.remove(num);
            } // Number appears first time, add to the set
            else {
                numbers.add(num);
            }
        }
        // Return the first (and only) element in the HashSet
        return numbers.iterator().next();
    }

    /**
     * Solution 2: Math solution
     * Expected sum (if all integers appeared twice): 2 * (sum_i=1^n distict_a_i)
     * Actual sum: sum_i=1^n a_i
     * Single number = Expected sum - Actual sum = 2 * (sum_i=1^n distict_a_i) - (sum_i=1^n a_i)
     * @param nums
     * @return
     */
    public static int singleNumber2(int[] nums){
        // Add all elements in the array into a HashSet, by converting array to list, then to HashSet.
        // Same task, with two one-liner solutions:
        // NOTE: Use of Collections.addAll and HashSet requires "import java.util.*;"
        // NOTE: In the first solution below, using "Arrays.asList(nums)" in the HashSet creation on RHS is not correct,
        // because nums is "int[]", whereas Arrays.asList(Collections) accepts generic arrays, e.g. Integer[].
        // Therefore, we convert int[] to integer[] using "Arrays.stream(nums).boxed().toArray(Integer[]::new))"
        // Solution 1
        Set<Integer> numberSet = new HashSet<Integer>(Arrays.asList(Arrays.stream(nums).boxed().toArray(Integer[]::new)));
        // Solution 2
        // Set<Integer> numberSet = Arrays.stream(nums).boxed().collect(Collectors.toSet());

        // NOTE: Use of IntStream requires "import java.util.stream.*;"
        int actualSum = IntStream.of(nums).sum();

        // Expected sum
        int expectedSum = 0;
        for(int num : numberSet){
            expectedSum += num;
        }
        expectedSum = expectedSum * 2;

        // Return single number
        int singleNumber = expectedSum - actualSum;
        return singleNumber;
    }

    /**
     * Solution 3: Bit Manipulation - XOR trick
     * If we XOR zero and some bit, it returns that bit: a^0 = a. If we XOR same bits, it will return 0: a^a = 0.
     * If all numbers except one are repeated twice, they will return 0: a_i ^ a_i = 0 for a_i which is repeated twice.
     * The single number will be XORed with 0, and it will be the result of XOR'in all numbers: 0 ^ a_j = a_j.
     * Time complexity: O(n)
     * Space complexity: O(1)
     * @param nums
     * @return
     */
    public static int singleNumber3(int[] nums) {
        int singleNumber = 0;
        for(int num : nums){
            singleNumber ^= num;
        }
        return singleNumber;
    }

    /**
     * Run a test case with all solutions
     * @param nums
     */
    public static void runTestCase(int[] nums){
        System.out.println("singleNumber1( " + Arrays.toString(nums) + " ): " + singleNumber1(nums));
        System.out.println("singleNumber2( " + Arrays.toString(nums) + " ): " + singleNumber2(nums));
        System.out.println("singleNumber3( " + Arrays.toString(nums) + " ): " + singleNumber3(nums));
    }

    public static void main(String[] args){
        // Test 1
        int[] numbers = new int[]{1,1,2,2,3,3,5};
        runTestCase(numbers);

        // Test 2: Array with only 1 element
        int[] numbers2 = new int[]{1};
        runTestCase(numbers2);
    }
}
