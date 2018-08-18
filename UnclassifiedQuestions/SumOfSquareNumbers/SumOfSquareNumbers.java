/**
 Given a non-negative integer c, your task is to decide whether there are two integers a and b such that a^2 + b^2 = c.

 Example 1:
 Input: 5
 Output: True
 Explanation: 1 * 1 + 2 * 2 = 5

 Example 2:
 Input: 3
 Output: False
 */

import java.util.*;

class SumOfSquareNumbers {
    /**
     * Solution 1
     * @param c
     * @return
     */
    public static boolean isSumOfSquares(int c) {
        double sqrtValue = Math.sqrt(c);
        for(int i = 0; i <= sqrtValue; i++){
            double rem = Math.sqrt(c - i*i);
            // If the floor value of square root is equal to the square root itself, the remaining number is a perfect square.
            if( Math.floor(rem) == rem ){
                return true;
            }
        }
        return false;
    }

    /**
     * Solution 2: Interesting HashSet based solution
     * @param c
     * @return
     */
    public static boolean isSumOfSquares2(int c) {
        HashSet<Integer> squareNumberSet = new HashSet<Integer>();
        for(int i = 0; i <= Math.sqrt(c); i++){
            squareNumberSet.add(i*i);
            int rem = c - i*i;
            if(squareNumberSet.contains(rem)){
                return true;
            }
        }
        return false;
    }

    /**
     * Solution 3: Using two pointers, one at the beginning and one at the end of the number range
     * @param c
     * @return
     */
    public static boolean isSumOfSquares3(int c) {
        int left = 0;
        int right = (int)Math.sqrt(c);
        while(left <= right) {
            int sum = (int)Math.pow(left, 2) + (int)Math.pow(right, 2);
            // If sum is smaller than c, increase left pointer to increase the sum
            if(sum < c) {
                left++;
            } // If sum is greater than c, decrease right pointer to decrease the sum
            else if(sum > c) {
                right--;
            } // If sum is equal to c, we have the sum of squares, and return true
            else {
                return true;
            }
        }
        // If there is no sum of squares returned in the while loop above, return false
        return false;
    }

    /**
     * Run all sum of square function variants for a given number
     * @param n
     */
    public static void runSumOfSquaresTests(int n){
        System.out.println(n + ": " + isSumOfSquares(n) + " (isSumOfSquares)");
        System.out.println(n + ": " + isSumOfSquares2(n) + " (isSumOfSquares2)");
        System.out.println(n + ": " + isSumOfSquares3(n) + " (isSumOfSquares3)");
    }

    public static void main(String[] args){
        // Test 1: True (10 = 1^2 + 3^2)
        int x = 10;
        runSumOfSquaresTests(x);

        // Test 2: True (8 = 2^2 + 2^2)
        x = 8;
        runSumOfSquaresTests(x);

        // Test 3: False (11)
        x = 11;
        runSumOfSquaresTests(x);
    }
}