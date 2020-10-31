/**
 Mike is a stock trader and makes a profit by buying and selling stocks. He buys a stock at a lower price and
 sells it at a higher price to book a profit. He has come to know the stock prices of a particular stock for n
 upcoming days in future and wants to calculate the maximum profit by doing the right transactions
 (single transaction = buying + selling). How can you maximize his profit.
 Note: A transaction starts after the previous transaction has ended. Two transactions can't overlap or run in parallel.

 The stock prices are given in the form of an array A for n days. Given the stock prices and a positive integer k, find
 and print the maximum profit Mike can make in at most k transactions.

 For example, 5-day stock prices are given as [12,5,10,7,17], and k=1. For one transaction, maximum profit is 12 when
 stock is purchased on day 2 and sold on day 5.
 */

/**
 * Links:
 * 1) http://www.geeksforgeeks.org/maximum-profit-by-buying-and-selling-a-share-at-most-k-times/
 * 2) https://discuss.leetcode.com/topic/4766/a-clean-dp-solution-which-generalizes-to-k-transactions
 * 3) Linear-time solution: https://stackoverflow.com/questions/9514191/maximizing-profit-for-given-stock-quotes
 * 4) Multiple solutions compared based on running time: https://github.com/mission-peace/interview/blob/master/src/com/interview/dynamic/StockBuySellKTransactions.java
 */

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class BuySellStockMaxKtransactions {

    // Running time: O(k * n^2), where n is A.length (number of days)
    static int getMaximumProfitFromStockInMaxKtransactions(int[] A, int k) {
        // Initialize profit table, of size (k+1) x (A.length).
        // Because number of transactions can be 0 to k, which are a list of (k+1) values,
        // whereas the number of days are fixed, A.length.
        int[][] profit = new int[k+1][A.length];

        // For day 0, profit is 0 for all k values
        for(int i = 0; i <= k; i++){
            profit[i][0] = 0;
        }

        // If k = 0, there is no transaction, and profit is 0.
        for(int j = 0; j < A.length; j++){
            profit[0][j] = 0;
        }

        // Fill the profit[k+1][A.length] table in bottom-up fashion
        // Start from indices = 1, because 0-indices of the table are filled in the for loops above
        // Logic: Let profit[i][j] represent maximum profit using at most i transactions up to day j (including day j).
        // Then, there are two cases based on what will happen on j-th day:
        // 1) There is no transaction on j-th day: In this case, profit is the profit of (j-1)-th day, profit[i][j-1].
        // 2) There is a transaction on j-th day: In this case, maximum profit is gained by selling on j-th day. In order
        // to sell shares on j-th day, we need to purchase it on any one of r \in [0, j–1] days. If we buy shares on j-th day
        // and sell it on r-th day, max profit will be ( (price[j] – price[r]) + profit[i-1][r] ), where r varies from 0
        // to (j-1). Here, price[j] is the amount we earn by selling on j-th day, price[r] is the amount we lose by buying
        // on r-th day, and profit[i-1][r] is the best we could have done with one less transaction till i-th day.
        // As a result, profit[i][j] will be maximum of these two options, described as follows:
        // profit[i][j] = max( profit[i][j-1], max_{r=0}^{r=j-1}(price[j] – price[r] + profit[i-1][r]) )
        for(int i = 1; i <= k; i++) {
            for(int j = 1; j < profit[0].length; j++) {
                int maxSoFar = Integer.MIN_VALUE;

                // For i transactions on j-th day, we iterate r in [0, j-1] range, to find the maximum value for option 2,
                // in which there is a transaction on j-th day, where maximum profit is gained by selling on j-th day.
                for(int r = 0; r < j; r++) {
                    maxSoFar = Math.max(maxSoFar, A[j] - A[r] + profit[i-1][r]);
                }

                // Get the maximum value returned by option 1) and 2)
                profit[i][j] = Math.max(profit[i][j-1], maxSoFar);
            }
        }
        return profit[k][A.length-1];
    }

    public static void main(String[] args) {
        // Test 1
        int[] stockValues1 = {10, 22, 5, 75, 65, 80};
        int k = 2;
        int result = getMaximumProfitFromStockInMaxKtransactions(stockValues1, k);
        System.out.println(result);

        // Test 2
        int[] stockValues2 = {10, 22, 5, 75, 65, 80, 100, 1000};
        k = 4;
        result = getMaximumProfitFromStockInMaxKtransactions(stockValues2, k);
        System.out.println(result);

        // Test 3
        // Special case: Mike cannot make any profit as selling price is decreasing day by day.
        // Hence, it is not possible to earn anything.
        int[] stockValues3 = {100, 90, 80, 50, 25};
        k = 5;
        result = getMaximumProfitFromStockInMaxKtransactions(stockValues3, k);
        System.out.println(result);
    }
}
