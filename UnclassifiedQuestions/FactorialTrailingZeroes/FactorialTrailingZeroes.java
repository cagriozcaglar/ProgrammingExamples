/**
 Given an integer n, return the number of trailing zeroes in n!.
 Note: Your solution should be in logarithmic time complexity.
 */

import java.util.*;

public class FactorialTrailingZeroes {
    /**
     * Solution 1, with an extra variable.
     * The 0 comes from 10, which is 2 x 5. And we need to account for all the products of 5 and 2. If we take all the
     * numbers with 5 as a factor, we'll have way more than enough even numbers to pair with them to get factors of 10.
     * Therefore, all we need to do is to count the number of factors of 5 in n!.
     * Number of trailing )s in n! = [n / 5^1] + [n / 5^2] + [n / 5^3] + ...
     * @param n
     * @return
     */
    public static int trailingZeroes1(int n) {
        int zeroCount = 0;
        int runner = n;
        while(runner > 0){
            runner = runner / 5;
            zeroCount += runner;
        }
        return zeroCount;
    }

    /**
     * Solution 2, with one less variable. Idea is the same as the one in the solution above.
     * The 0 comes from 10, which is 2 x 5. And we need to account for all the products of 5 and 2. If we take all the
     * numbers with 5 as a factor, we'll have way more than enough even numbers to pair with them to get factors of 10.
     * Therefore, all we need to do is to count the number of factors of 5 in n!.
     * Number of trailing )s in n! = [n / 5^1] + [n / 5^2] + [n / 5^3] + ...
     * @param n
     * @return
     */
    public static int trailingZeroes2(int n) {
        int zeroCount = 0;
        // No "runner" variable as in the solution above, we just use "n" itself as the runner
        while(n > 0){
            n = n / 5;
            zeroCount += n;
        }
        return zeroCount;
    }

    /**
     * Runner of all solutions
     * @param n
     */
    public static void runTestCases(int n){
        // Timers for profiling
        long startTime;
        long endTime;

        // Solution 1
        startTime = System.nanoTime();
        int zeroCount1 = trailingZeroes1(n);
        endTime =  System.nanoTime();
        System.out.println("trailingZeroes1(" + n + "): " + zeroCount1 + " (Time: " + (endTime - startTime) + " ns)");

        // Solution 2
        startTime = System.nanoTime();
        int zeroCount2 = trailingZeroes2(n);
        endTime =  System.nanoTime();
        System.out.println("trailingZeroes2(" + n + "): " + zeroCount2 + " (Time: " + (endTime - startTime) + " ns)");
    }

    public static void main(String[] args){
        // Test case 1: n = 5, n! = 120
        int n = 5;
        runTestCases(n);

        // Test case 2: n = 4, n! = 24
        n = 4;
        runTestCases(n);

        // Test case 3: n = 10
        n = 10;
        runTestCases(n);

        // Test case 4: n = 25
        n = 25;
        runTestCases(n);
    }
}