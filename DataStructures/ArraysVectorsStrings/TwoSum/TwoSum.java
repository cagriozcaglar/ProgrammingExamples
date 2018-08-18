/**
 Given an array of integers, return indices of the two numbers such that they add up to a specific target.
 You may assume that each input would have exactly one solution, and you may not use the same element twice.

 Example: Given nums = [2, 7, 11, 15], target = 9,
 Because nums[0] + nums[1] = 2 + 7 = 9, return [0, 1].
 */

import java.util.*;

public class TwoSum {

    /**
     * TwoSum function: returns the index pair of an array "nums", where array values at these indices are equal to "target"
     * Iterate over the array, insert elements to a HashMap mapping values to array index. If current element's complement
     * already exists in the map, we have found a solution and immediately return.
     * Running time: O(n)
     * Space complexity: O(n)
     * @param nums
     * @param target
     * @return
     */
    public static int[] twoSum(int[] nums, int target) {
        int[] indices = new int[2];
        HashMap<Integer, Integer> numberIndexMap = new HashMap<Integer, Integer>();
        for(int i = 0; i < nums.length; i++){
            int complement = target - nums[i];
            if(numberIndexMap.containsKey(complement)){
                indices[0] = numberIndexMap.get(complement);
                indices[1] = i;
                break;
            }
            numberIndexMap.put(nums[i], i);
        }
        return indices;
    }

    /**
     * TwoSum function - shorter: Same logic as above, but shorter in terms of number of lines.
     * Also has exception handling, in case where there is no pair of indices that sum up to target
     * Running time: O(n)
     * Space complexity: O(n)
     * @param nums
     * @param target
     * @return
     */
    public static int[] twoSumShorter(int[] nums, int target) {
        Map<Integer, Integer> numberIndexMap = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (numberIndexMap.containsKey(complement)) {
                return new int[] { numberIndexMap.get(complement), i };
            }
            numberIndexMap.put(nums[i], i);
        }
        throw new IllegalArgumentException("No two sum solution");
    }

    // Print indices of array
    public static void printIndices(int[] indices){
        System.out.println("[" + indices[0] + ", " + indices[1] + "]");
    }

    // Main
    public static void main(String[] args){
        // Test 1
        int[] array = {3, 4, 5, 6};
        int target = 8;
        // twoSum function: no exception handling
        int[] values = twoSum(array, target);
        printIndices(values);
        // twoSumShorter function: with exception handling
        int[] values2 = twoSumShorter(array, target);
        printIndices(values2);

        // Test 2: Case where there is no pair of indices that sum up to target
        int[] array2 = {3, 9, 2, 1};
        target = 15;
        // twoSum function: no exception handling: Returns indices [0,0], which are default values of initialized int array
        int[] values3 = twoSum(array2, target);
        // twoSumShorter function: with exception handling: Returns Illegal ArgumentException
        printIndices(values3);
        // The following line will return an exception. Remove to proceed to next operations
        //int[] values4 = twoSumShorter(array2, target);
        //printIndices(values4);

        // TwoSum 2: sorted
        // Test 1:
        int[] array3 = {3, 4, 5, 6};
        target = 8;
        // twoSum function: no exception handling
        int[] values5 = TwoSum2sorted.twoSum2sorted(array3, target);
        printIndices(values5);

        // Test 2: Case where there is no pair of indices that sum up to target
        int[] array4 = {3, 9, 2, 1};
        target = 15;
        // twoSum function: with exception handling: Returns Illegal ArgumentException
        //int[] values6 = TwoSum2sorted.twoSum2sorted(array4, target);
        //printIndices(values6);

        // Two Sum 3: Data Structure Design
        TwoSum3DataStructureDesign.add(1);
        TwoSum3DataStructureDesign.add(3);
        TwoSum3DataStructureDesign.add(5);
        System.out.println(TwoSum3DataStructureDesign.find(4));
        System.out.println(TwoSum3DataStructureDesign.find(7));
    }
}

class TwoSum2sorted {
    /**
     * TwoSum II - sorted: Given an array of integers that is already sorted in ascending order, find two numbers such
     * that they add up to a specific target number. The function twoSum should return indices of the two numbers such
     * that they add up to the target, where index1 must be less than index2. Please note that your returned answers
     * (both index1 and index2) are not zero-based. You may assume that each input would have exactly one solution and
     * you may not use the same element twice.
     *
     * Input: numbers={2, 7, 11, 15}, target=9
     * Output: index1=1, index2=2
     *
     * Running time: O(n)
     * Note: A binary-search based solution is also possible, and the running time will be O(log(n)) on average.
     *
     * @param numbers
     * @param target
     * @return
     */
    public static int[] twoSum2sorted(int[] numbers, int target) {
        int first = 0;
        int last = numbers.length - 1;
        while (first != last) {
            int sum = numbers[first] + numbers[last];
            // If below target, increase first pointer
            if (sum < target) {
                first++;
            } else if (sum > target) {
                last--;
            } else if (sum == target) {
                // Return indices are 1-based, although array indices are 0-based
                int[] solution = {first + 1, last + 1};
                return solution;
            }
        }
        throw new IllegalArgumentException("No solution for TwoSum 2 problem");
    }
}

class TwoSum3DataStructureDesign {
    /**
     * Design and implement a TwoSum class. It should support the following operations: add and find.
     * 1) add - Add the number to an internal data structure.
     * 2) find - Find if there exists any pair of numbers which sum is equal to the value.

     * For example,
     * add(1);
     * add(3);
     * add(5);
     * find(4) -> true
     * find(7) -> false
     */
    private static HashMap<Integer, Integer> countMap = new HashMap<Integer, Integer>();

    public static void add(int number){
        if(countMap.containsKey(number)){
            countMap.put(number, countMap.get(number)+1);
        } else {
            countMap.put(number, 1);
        }
    }

    public static boolean find(int number){
        for(int value : countMap.keySet()){
            int complement = number - value;
            if(countMap.containsKey(complement)) {
                // Careful here: Values of (complement, value) pair can be the same, although they sum to "number".
                // Check if "complement == value" && elements.get(value) < 2. If so, there is only one such number,
                // and no pair of values can be found, and continue. Otherwise, return true.
                // This is also why a HashSet as the main data structure is not enough.
                if(complement == value && countMap.get(value) < 2){
                    continue;
                }
                return true;
            }
        }
        return false;
    }
}