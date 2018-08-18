/**
 Let f(x) be the number of zeroes at the end of x!. (Recall that x! = 1 * 2 * 3 * ... * x, and by convention, 0! = 1.)
 For example, f(3) = 0 because 3! = 6 has no zeroes at the end, while f(11) = 2 because 11! = 39916800 has 2 zeroes at
 the end. Given K, find how many non-negative integers x have the property that f(x) = K.

 Example 1:
 Input: K = 0
 Output: 5
 Explanation: 0!, 1!, 2!, 3!, and 4! end with K = 0 zeroes.

 Example 2:
 Input: K = 5
 Output: 0
 Explanation: There is no x such that x! ends in K = 5 zeroes.

 Note: K will be an integer in the range [0, 10^9].
 */

/**
 * Link: https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/discuss/117821/Four-binary-search-solutions-based-on-different-ideas
 // Similar to https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/discuss/117584/JAVA-Binary-search
 // Other solutions:
 // 1. K zeroes at least: https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/discuss/117642/C++-4ms-binary-search-solution-less-than-20-lines
 // 2. Upper / lower bound: https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/discuss/117581/Binary-Search-in-Java-find-Upper-bound-and-lower-bound
 // 3. Find range: https://leetcode.com/problems/preimage-size-of-factorial-zeroes-function/discuss/117619/Using-Binary-Search-Java-Solution
 */


public class PremiageSizeOfFactorialZeroes {
    class Solution {
        public int preimageSizeFZF(int K) {
            if(K==0) {return 5;}
            int low = 0;
            int high = Integer.MAX_VALUE;
            int mid = low + (high - low) / 2;
            int midNumber;
            // Binary search
            while(low < high) {
                mid = low + (high - low) / 2;
                midNumber = numberOfZeroes(mid);
                if (midNumber == K) {
                    return 5;
                } else if( midNumber > K) {
                    high = mid-1;  // Left half
                } else if (midNumber < K) {
                    low = mid+1;   // Right half
                }
            }
            return 0;
        }

        public int numberOfZeroes(int number) {
            return ( (number<5) ? 0 : number/5 + numberOfZeroes(number/5));
        }
    }
}