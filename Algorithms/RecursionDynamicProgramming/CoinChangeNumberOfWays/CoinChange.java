/**
 You have m types of coins available in infinite quantities where the value of each coin is given in
 the array C=[c_0,c_1,...,c_(m-1)]. Can you determine the number of ways of making change for n units
 using the given types of coins? For example, if m=4, and C=[8,3,1,2], we can make change for n=3 units
 in three ways: {1,1,1}, {1,2}, and {3}.

 Given n, m, and C, print the number of ways to make change for n units using any number of coins having the values given in C.

 Hints:
 1) Solve overlapping subproblems using Dynamic Programming (DP): You can solve this problem recursively,
 but will not pass all the test cases without optimizing to eliminate the overlapping subproblems. Think of
 a way to store and reference previously computed solutions to avoid solving the same subproblem multiple times.
 2) Consider the degenerate cases:
   2.1) How many ways can you make change for 0 cents?
   2.2) How many ways can you make change for >0 cents if you have no coins?
 3) If you're having trouble defining your solutions store, then think about it in terms of the base case (n=0).
 4) The answer may be larger than a 32-bit integer (Hence you may need to return a long (64-bit integer),
 instead of int (32-bit integer)).
 */

/**
 * Example links:
 * 1) http://algorithms.tutorialhorizon.com/dynamic-programming-coin-change-problem/
 * 2) http://www.geeksforgeeks.org/dynamic-programming-set-7-coin-change/
 * 3) Cracking the coding algorithm, 6th edition, question 8.11, page 374
 */

/**
 * Related question: Coin change with minimum number of coins: http://ace.cs.ohiou.edu/~razvan/courses/cs4040/lecture19.pdf
 */

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class CoinChange {

    // Recursive solution
    static long getWaysRecursive(long target, long[] coins, int coinIndex) {
        // Base case 1: We ran out of target money.
        // If target is reached, there is only one solution, which is to add no other coins.
        if(target == 0){
            return 1;
        }
        // Base case 2: We ran out of coins (passed previous condition), but not target money.
        // If coinIndex is outside the range of coins list, there is no solution.
        if(coinIndex >= coins.length){
            return 0;
        }
        // Note: The order of two if conditions above matter. If you switch the order, the result will be incorrect.
        // So, first, check if the target is reached, and second, check if coinIndex is within bounds.
        long coinAmount = coins[coinIndex];
        long ways = 0;

        // Use as many cointAmount values as possible to reach target
        // For the rest, make a recursive call for the next index of coins array
        for(int i = 0; i * coinAmount <= target; i++) {
            long remainingToTarget = target - (i * coinAmount);
            ways += getWaysRecursive(remainingToTarget, coins, (coinIndex+1));
        }
        return ways;
    }

    // Dynamic Programming solution
    static long getWaysDynamicProgramming(long target, long[] coins, int coinIndex, HashMap<String, Long> memo) {
        // Base case 1: We ran out of target money.
        // If target is reached, there is only one solution, which is to add no other coins.
        if(target == 0){
            return 1;
        }
        // Base case 2: We ran out of coins (passed previous condition), but not target money.
        // If coinIndex is outside the range of coins list, there is no solution.
        if(coinIndex >= coins.length){
            return 0;
        }
        // Note: The order of two if conditions above matter. If you switch the order, the result will be incorrect.
        // So, first, check if the target is reached, and second, check if coinIndex is within bounds.

        // ************** Start of Additional code block compared to recursive solution **************
        // Check if solution exists, if so, return
        // Any key will do fine, as long as it is unique.
        // Note: we added "_" to prevent duplicates. E.g. "291" -> {"29", "1"} or {"2", "91"}
        String key = target + "_" + coinIndex;
        if(memo.containsKey(key)){
            return memo.get(key);
        }
        // ************** End of Additional code block compared to recursive solution ****************

        long coinAmount = coins[coinIndex];
        long ways = 0;

        // Use as many cointAmount values as possible to reach target
        // For the rest, make a recursive call for the next index of coins array
        for(int i = 0; i * coinAmount <= target; i++) {
            long remainingToTarget = target - (i * coinAmount);
            // ************** Start of DP-call instead of recursive call **************
            ways += getWaysDynamicProgramming(remainingToTarget, coins, (coinIndex+1), memo);
            // ************** End of DP-call instead of recursive call ****************
        }
        // ************** Start of Additional code block compared to recursive solution **************
        // Save the current solution in the memo
        memo.put(key, ways);
        // ************** End of Additional code block compared to recursive solution ****************

        return ways;
    }

    static void getAllWays(long target, long[] coins){
        // Timers for profiling
        long startTime;
        long endTime;

        // Recursive solution
        startTime = System.nanoTime();
        long recursiveWays = getWaysRecursive(target, coins, 0);
        endTime =  System.nanoTime();
        System.out.println("Number of ways generated by recursive solution: " + recursiveWays);
        System.out.println("Time for recursive solution: " + (endTime - startTime));

        // Dynamic Programming solution
        startTime = System.nanoTime();
        long dynamicProgrammingWays = getWaysDynamicProgramming(target, coins, 0, new HashMap<String, Long>());
        endTime =  System.nanoTime();
        System.out.println("Number of ways generated by recursive solution: " + dynamicProgrammingWays);
        System.out.println("Time for dynamic programming solution: " + (endTime - startTime) + "\n");
    }

    public static void main(String[] args) {
        /**
         * Outcome: For small coin values, memoization is an overhead for dynamic programming solution, so the recursive
         * solution is faster for small inputs / coin values. For large coin values, search space increases (2^n), and memoization
         * in dynamic programming saves time, and therefore, dynamic programming solution becomes much faster than recursive
         * solution.
         */

        // Test 1
        int n = 4;
        long[] coins1 = {1, 2, 3};
        getAllWays(n, coins1);
        /*
        Number of ways: 4
        Recursive:             4,595 ns
        Dynamic programming: 538,243 ns
        */

        // Test 2
        n = 10;
        long[] coins2 = {2, 3, 5, 6};
        getAllWays(n, coins2);
        /*
        Number of ways: 5
        Recursive:            6,057 ns
        Dynamic programming: 91,954 ns
        */

        // Test 3
        n = 100;
        long[] coins3 = {2, 3, 5, 6};
        getAllWays(n, coins3);
        /*
        Number of ways: 1163
        Recursive:             888,360 ns
        Dynamic programming: 3,817,729 ns
        */

        // Test 4
        n = 1000;
        long[] coins4 = {2, 3, 5, 6};
        getAllWays(n, coins4);
        /*
        Number of ways: 948,293
        Recursive:            804,397,007 ns
        Dynamic programming:   72,303,683 ns
        */

        // Test 5
        n = 2000;
        long[] coins5 = {2, 3, 5, 6};
        getAllWays(n, coins5);
        /*
        Number of ways: 7,496,697
        Recursive:            12,104,041,809 ns
        Dynamic programming:     112,009,974 ns
        */
    }
}