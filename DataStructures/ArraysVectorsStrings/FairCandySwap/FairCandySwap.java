/**
 Alice and Bob have candy bars of different sizes: A[i] is the size of the i-th bar of candy that Alice has, and B[j] is
 the size of the j-th bar of candy that Bob has.
 Since they are friends, they would like to exchange one candy bar each so that after the exchange, they both have the
 same total amount of candy.  (The total amount of candy a person has is the sum of the sizes of candy bars they have.)
 Return an integer array ans where ans[0] is the size of the candy bar that Alice must exchange, and ans[1] is the size
 of the candy bar that Bob must exchange.
 If there are multiple answers, you may return any one of them.  It is guaranteed an answer exists.

 Example 1:
 Input: A = [1,1], B = [2,2]
 Output: [1,2]

 Example 2:
 Input: A = [1,2], B = [2,3]
 Output: [1,2]

 Example 3:
 Input: A = [2], B = [1,3]
 Output: [2,3]

 Example 4:
 Input: A = [1,2,5], B = [2,4]
 Output: [5,4]

 Note:
 => 1 <= A.length <= 10000
 => 1 <= B.length <= 10000
 => 1 <= A[i] <= 100000
 => 1 <= B[i] <= 100000
 => It is guaranteed that Alice and Bob have different total amounts of candy.
 => It is guaranteed there exists an answer.
 */

import java.util.*;
import java.util.stream.*;

public class FairCandySwap {

    /**
     * Similar to two-sum problem.
     * Expected difference A[i] - B[j] has to be equal to (sumA - sumB) / 2. (Divided by two, because the swap will
     * decrease / increase sumB by expected difference, and increase / decrease sumA by expected difference).
     * Then, add elements of A to a HashSet<Integer>. Then iterate over elements of B, and if there exists a b in B
     * such that HashSet<Integer> contains (b + expectedDiff), then, return the pair (b, b+expectedDiff).
     *
     * Example: https://leetcode.com/problems/fair-candy-swap/discuss/161269/C++JavaPython-Straight-Forward
     *
     * @param A
     * @param B
     * @return
     */
    public static int[] fairCandySwap(int[] A, int[] B) {
        // NOTE: Short way to calculate the sum of an integer array
        // https://stackoverflow.com/questions/4550662/how-do-you-find-the-sum-of-all-the-numbers-in-an-array-in-java
        int sumA = IntStream.of(A).sum();
        int sumB = IntStream.of(B).sum();

        // Explanation of this condition: if (sumA-sumB) is odd, there is no way to balance sumA and sumB by element swapping
        if((sumA-sumB)%2 != 0) {
            return new int[]{};
        }

        // This is calculated only if (sumA-sumB) is even
        int expectedDiff = (sumA - sumB) / 2;
        // System.out.println("Expected diff: " + expectedDiff);

        // Keep elements of A in a HashSet<Integer>
        HashSet<Integer> lookup = new HashSet<Integer>();
        for(int a : A) {
            lookup.add(a);
        }

        // Iterate over elements of B, and if there exists a b in B such that "lookup" contains (b + expectedDiff),
        // return the pair (b, b+expectedDiff).
        for(int b : B) {
            // Explanation of second condition: if (sumA-sumB) is odd, (sumA-sumB)/2 is rounded down to expectedDiff.
            // To prevent this, we check if "expectedDiff * 2 == (sumA-sumB)", meaning, if (sumA-sumB) was even.
            if(lookup.contains(b + expectedDiff)) {
                return new int[]{b + expectedDiff, b}; // CAREFUL: Be careful about the order of elements
            }
        }

        // If no pair found, return empty array
        return new int[]{};
    }

    /**
     * Sorting
     *
     * Time complexity: O(n*log(n))
     *
     * @param A
     * @param B
     * @return
     */
    public static int[] fairCandySwap2(int[] A, int[] B) {
        // NOTE: Short way to calculate the sum of an integer array
        // https://stackoverflow.com/questions/4550662/how-do-you-find-the-sum-of-all-the-numbers-in-an-array-in-java
        int sumA = IntStream.of(A).sum();
        int sumB = IntStream.of(B).sum();

        // Explanation of this condition: if (sumA-sumB) is odd, there is no way to balance sumA and sumB by element swapping
        if((sumA-sumB)%2 != 0) {
            return new int[]{};
        }

        // This is calculated only if (sumA-sumB) is even
        int expectedDiff = (sumA - sumB) / 2;
        // System.out.println("Expected diff: " + expectedDiff);

        // Sort both arrays in-place
        Arrays.sort(A);
        Arrays.sort(B);
        // System.out.println("First Array: " + Arrays.toString(A));
        // System.out.println("Second Array: " + Arrays.toString(B));

        for(int i = 0, j = 0; i < A.length && j < B.length; ) {
            int currentDiff = A[i] - B[j];
            // System.out.println("Current diff: " + currentDiff);
            if(currentDiff < expectedDiff) {
                // System.out.println("First if");
                i++;
            } else if(currentDiff > expectedDiff) {
                // System.out.println("Else if");
                j++;
            } else {
                // System.out.println("Else");
                return new int[]{A[i], B[j]};
            }
        }

        // If no pair available, return empty array
        return new int[]{};
    }

    /**
     * Run a test case with all solutions
     * @param A
     * @param B
     */
    public static void runTestCases(int[] A, int[] B) {
        // fairCandySwap solution with HashSet (O(n))
        System.out.println("fairCandySwap solution with HashSet (O(n)) for arrays " +
                            Arrays.toString(A) + " and " + Arrays.toString(B) + ": " +
                            Arrays.toString(fairCandySwap(A,B)));
        // fairCandySwap solution with array sorting (O(n*log(n)))
        System.out.println("fairCandySwap solution with array sorting (O(n*log(n))) for arrays " +
                            Arrays.toString(A) + " and " + Arrays.toString(B) + ": " +
                            Arrays.toString(fairCandySwap2(A,B)));
        System.out.println();
    }

    public static void main(String[] args) {
        /**
         * Test 1:
         fairCandySwap solution with HashSet (O(n)) for arrays [1, 1] and [2, 2]: [1, 2]
         fairCandySwap solution with array sorting (O(n*log(n))) for arrays [1, 1] and [2, 2]: [1, 2]
         */
        int[] A1 = new int[]{1, 1};
        int[] B1 = new int[]{2, 2};
        runTestCases(A1, B1);

        /**
         * Test 2:
         // Note: Results from two methods are different, that's because the second method sorts the arrays,
         // and as a result, the first pair that makes sumA == sumB changes.
         fairCandySwap solution with HashSet (O(n)) for arrays [2, 1] and [3, 2]: [2, 3]
         fairCandySwap solution with array sorting (O(n*log(n))) for arrays [2, 1] and [3, 2]: [1, 2]
         */
        int[] A2 = new int[]{2, 1}; // SumA: 3
        int[] B2 = new int[]{3, 2}; // SumB: 5
        runTestCases(A2, B2);

        /**
         * Test 3:
         fairCandySwap solution with HashSet (O(n)) for arrays [2] and [1, 3]: [2, 3]
         fairCandySwap solution with array sorting (O(n*log(n))) for arrays [2] and [1, 3]: [2, 3]
         */
        int[] A3 = new int[]{2};
        int[] B3 = new int[]{1, 3};
        runTestCases(A3, B3);

        /**
         * Test 4:
         fairCandySwap solution with HashSet (O(n)) for arrays [1, 2, 5] and [2, 4]: [5, 4]
         fairCandySwap solution with array sorting (O(n*log(n))) for arrays [1, 2, 5] and [2, 4]: [5, 4]
         */
        int[] A4 = new int[]{1, 2, 5};
        int[] B4 = new int[]{2, 4};
        runTestCases(A4, B4);

        /**
         * Test 5:
         fairCandySwap solution with HashSet (O(n)) for arrays [1, 2, 3, 4] and [1, 2, 4]: [2, 1]
         fairCandySwap solution with array sorting (O(n*log(n))) for arrays [1, 2, 3, 4] and [1, 2, 4]: [2, 1]
         */
        int[] A5 = new int[]{1, 2, 3, 4};
        int[] B5 = new int[]{1, 2, 4};
        runTestCases(A5, B5);
    }
}