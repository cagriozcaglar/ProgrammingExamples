/**
 The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

 Now your job is to find the total Hamming distance between all pairs of the given numbers.

 Example:
 Input: 4, 14, 2

 Output: 6

 Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
 showing the four bits relevant in this case). So the answer will be:
 HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
 Note:
 Elements of the given array are in the range of 0 to 10^9
 Length of the array will not exceed 10^4.
 */

import java.util.*;

public class TotalHammingDistance {
    /**
     * For each bit position 1-32 in a 32-bit integer, count the number of integers with that bit set to 0 and set to 1.
     * Then, if there are n integers in the array and k of them have a particular bit set and (n-k) do not, then that bit
     * contributes k*(n-k) hamming distance to the total. If there are numBit = 32 bits, then the sum is:
     * sum_{bit=1}^{bit=32} k_i * (n-k_i)
     *
     * Time complexity: O(n*d), where d is the number of bits in integers (e.g. 32 for 32-bit integers). Variable d is
     * constant in most cases, therefore, the running time is asympototically O(n*d) ~= O(n).
     *
     * @param numbers
     * @return
     */
    public static int getTotalHammingDistance(int[] numbers) {
        int totalHammingDistance = 0;
        int numberCount = numbers.length;
        int numberOfBits = 32;

        // Iterate over each bit
        for(int bit = 0; bit < numberOfBits; bit++){
            // Initialize number of 1's for the current bit
            int numberOfOnes = 0;
            // Iterate over all numbers for the current bit
            for(int i = 0; i < numberCount; i++){
                numberOfOnes += numbers[i] & 1; // RHS: 0 or 1 depending on the last bit
                numbers[i] = numbers[i] >> 1;   // shift by one bit. NOTE: Short form: numbers[i] >>= 1
            }
            totalHammingDistance += numberOfOnes * (numberCount - numberOfOnes);
        }
        return totalHammingDistance;
    }

    /**
     * Run all test cases
     */
    public static void runTestCases(int[] numbers){
        System.out.println("Total Hamming distance of elements in " + Arrays.toString(numbers) + " is: " +
                            getTotalHammingDistance(numbers) + ".");
    }

    public static void main(String[] args) {
        // Test 1
        int[] numbers = {0, 1, 2, 3}; // 00, 01, 10, 11 => Bit 0: 2*2=4, Bit 1: 2*2=4, Total = 4+4 = 8
        runTestCases(numbers);

        // Test 2
        int[] numbers2 = {4, 14, 2}; // HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
        runTestCases(numbers2);
    }
}