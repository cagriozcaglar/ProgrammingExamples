/**
 You are climbing a stair case. It takes n steps to reach to the top.
 Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
 Note: Given n will be a positive integer.
*/

import java.util.*; // For HashMap

class ClimbingStairs {
    /**
     * Solution 1: Dynamic Programming, with an array
     *
     * The problem seems to be a dynamic programming one. To get the solution incrementally:
     * Base cases:
     *  1) if n <= 0, then the number of ways should be zero.
     *  2) if n == 1, then there is only way to climb the stair.
     *  3) if n == 2, then there are two ways to climb the stairs. One solution is one step by another; the other one is
     * two steps at one time.
     *
     * The key intuition to solve the problem is that given a number of stairs n, if we know the number ways to get to
     * the points [n-1] and [n-2] respectively, denoted as n1 and n2, then the total ways to get to the point [n] is
     * n1 + n2. Because from the [n-1] point, we can take one single step to reach [n]. And from the [n-2] point, we
     * could take two steps to get there. There is no overlapping between these two solution sets, because they differ in
     * the final step. Note that this is a bottom-up approach with memoization. It is also basically Fibonacci series,
     * shifted by one position (F(1) = 1, F(2) = 2, whereas Fib(1)=1, Fib(2)=1, Fib(3)=2, ...).
     *
     * Given the above intuition, one can construct an array where each node stores the solution for each number n. Or
     * if we look at it closer, it is clear that this is basically a Fibonacci number, with the starting numbers as 1
     * and 2, instead of 1 and 1.
     *
     * Time complexity: O(n)
     * @param n
     * @return
     */
    public static int climbStairs1(int n) {
        // Base case
        if(n == 0 || n == 1 || n == 2){
            return n;
        }

        // Keep results in an array
        int[] results = new int[n];

        // Add base case results to result array
        results[0] = 1;  // F(1) = 1 (index off by 1)
        results[1] = 2;  // F(2) = 2 (index off by 1)

        // Fill the array with results
        for(int i = 2; i < n; i++){
            results[i] = results[i-1] + results[i-2];
        }

        // Return the number of ways to climb n stairs
        return results[n-1];  // (index off by 1)
    }

    /**
     * Solution 2: Dynamic Programming, with a HashMap
     * Same explanation as above, except we use a HashMap instead of an array to save results
     * Time Complexity: O(n)
     * @param n
     * @return
     */
    public static int climbStairs2(int n) {
        HashMap<Integer, Integer> results = new HashMap<Integer, Integer>();
        results.put(0,1);  // F(0) = Fib(1) = 1 (index off by 1)
        results.put(1,1);  // F(1) = Fib(2) = 1 (index off by 1)
        for(int i = 2; i <= n; i++ ){
            results.put(i, results.get(i-1)+ results.get(i-2));
        }

        return results.get(n);
    }

    /**
     * Solution 3: Recursive solution
     *
     * For base case (n = 1 or 2), return n. ( Fib(1) = 1, Fib(2) = 2 ).
     * For n > 2: Fib(n) = Fib(n-1) + Fib(n-2). But there are a lot of repeated calls in this expansion.
     * Not suggested, as the search space grows exponentially, and becomes intractable when n is large.
     *
     * Time complexity: O(2^n)
     *
     * @param n
     * @return
     */
    public static int climbStairs3(int n) {
        if(n == 1 || n == 2) {
            return n;
        } else {
            return climbStairs3(n-1) + climbStairs3(n-2);
        }
    }

    /**
     * Solution 4: Use the closed-form calculation of Fibonacci number
     *
     * Fib(n) = ( p^n - (-p)^(-n) ) / sqrt(5),
     * where p = (1 + sqrt(5)) / 2
     *
     * Note that we need to return (n+1)-th Fibonacci number based on this formula, because the values are shifted by 1:
     * E.g. (Ways(1) = 1, Ways(2) = 2, whereas Fib(1)=1, Fib(2)=1, Fib(3)=2, ...).
     * See these pages for more details:
     * 1) Wikipedia: https://en.wikipedia.org/wiki/Fibonacci_number#Closed-form_expression
     * 2) Leetcode: https://discuss.leetcode.com/topic/30638/using-the-fibonacci-formular-to-get-the-answer-directly
     *
     * Time complexity: O(1)
     *
     * @param n
     * @return
     */
    public static int climbStairs4(int n) {
        n++;
        double sqrt5 = Math.sqrt(5);
        double phi = (1 + sqrt5) / 2;
        double result = ( Math.pow(phi, n) - Math.pow(-phi, -n) ) /sqrt5;
        return (int)result;
    }

    /**
     * Run a test case with all solutions
     * @param n
     */
    public static void runTestCase(int n){
        // Timers for profiling
        long startTime;
        long endTime;

        // Solution 1
        startTime = System.nanoTime();
        long climbStairs1Ways = climbStairs1(n);
        endTime =  System.nanoTime();
        System.out.println("climbStairs1(" + n + "): " + climbStairs1(n) + " (Time: " + (endTime - startTime) + " ns.)");

        // Solution 2
        startTime = System.nanoTime();
        long climbStairs2Ways = climbStairs2(n);
        endTime =  System.nanoTime();
        System.out.println("climbStairs2(" + n + "): " + climbStairs2(n) + " (Time: " + (endTime - startTime) + " ns.)");

        // Solution 3
        startTime = System.nanoTime();
        long climbStairs3Ways = climbStairs3(n);
        endTime =  System.nanoTime();
        System.out.println("climbStairs3(" + n + "): " + climbStairs3(n) + " (Time: " + (endTime - startTime) + " ns.)");

        // Solution 4
        startTime = System.nanoTime();
        long climbStairs4Ways = climbStairs4(n);
        endTime =  System.nanoTime();
        System.out.println("climbStairs4(" + n + "): " + climbStairs4(n) + " (Time: " + (endTime - startTime) + " ns.)");

        System.out.println();
    }

    /**
     * See output at the end of the function. Observations here:
     * 1) For small values of n, the gap between the time it takes to complete dynamic programming solution and recursive
     * solution is small. For large values of n, the gap increases exponentially.
     * 2) Interestingly, for small and large values of n, closed-form expression solution 4 is slighly slower than
     * dynamic programming solution 1. The gap is large when n is small, and the gap is small when n is large.
     * 3) Comparison of two dynamic programming solutions 1 (with array) and 2 (with HashMap): The DP solution with array
     * is always faster than the DP solution with HashMap, due to hash code calculation and possible use of HashMap
     * collision handling methods such as chaining or probing.
     * 4) Never, ever use the recursive solution in cases where the search space uses repeated calls, as in here.
     * 5) Observation valid for all methods: The number easily overflows when n > 40. Therefore, when n=50, the functions
     * return a negative integer.
     *
     * @param args
     */
    public static void main(String[] args){
        // Test 1
        int n = 1;
        runTestCase(n);

        // Test 2
        n = 2;
        runTestCase(n);

        // Test 3
        n = 4;
        runTestCase(n);

        // Test 4
        n = 10;
        runTestCase(n);

        // Test 5
        n = 20;
        runTestCase(n);

        // Test 6
        n = 30;
        runTestCase(n);

        // Test 7
        n = 40;
        runTestCase(n);

        // Test 8
        n = 50;
        runTestCase(n);
    }
    /** Output:
     climbStairs1(1): 1 (Time: 1434 ns.)
     climbStairs2(1): 1 (Time: 49328 ns.)
     climbStairs3(1): 1 (Time: 2100 ns.)
     climbStairs4(1): 1 (Time: 22064 ns.)

     climbStairs1(2): 2 (Time: 413 ns.)
     climbStairs2(2): 2 (Time: 7206 ns.)
     climbStairs3(2): 2 (Time: 290 ns.)
     climbStairs4(2): 2 (Time: 1071 ns.)

     climbStairs1(4): 5 (Time: 995 ns.)
     climbStairs2(4): 5 (Time: 10660 ns.)
     climbStairs3(4): 5 (Time: 918 ns.)
     climbStairs4(4): 5 (Time: 1064 ns.)

     climbStairs1(10): 89 (Time: 1497 ns.)
     climbStairs2(10): 89 (Time: 31495 ns.)
     climbStairs3(10): 89 (Time: 8224 ns.)
     climbStairs4(10): 89 (Time: 910 ns.)

     climbStairs1(20): 10946 (Time: 1885 ns.)
     climbStairs2(20): 10946 (Time: 44039 ns.)
     climbStairs3(20): 10946 (Time: 643522 ns.)
     climbStairs4(20): 10946 (Time: 1130 ns.)

     climbStairs1(30): 1346269 (Time: 2024 ns.)
     climbStairs2(30): 1346269 (Time: 103027 ns.)
     climbStairs3(30): 1346269 (Time: 3130304 ns.)
     climbStairs4(30): 1346269 (Time: 3247 ns.)

     climbStairs1(40): 165580141 (Time: 2889 ns.)
     climbStairs2(40): 165580141 (Time: 100332 ns.)
     climbStairs3(40): 165580141 (Time: 311750193 ns.)
     climbStairs4(40): 165580141 (Time: 3317 ns.)

     climbStairs1(50): -1109825406 (Time: 12578 ns.)
     climbStairs2(50): -1109825406 (Time: 119624 ns.)
      ====> It gets stuck here, because it runs forever for the recursive solution 3.
     */
}