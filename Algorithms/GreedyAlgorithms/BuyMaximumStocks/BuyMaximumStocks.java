/**
 In a stock market, there is a product with its infinite stocks. The stock prices are given for n days, where arr[i]
 denotes the price of the stock on the i-th day. There is a rule that a customer can buy at most i stocks on the i-th day.
 If the customer has an amount of k dollars initially, find out the maximum number of stocks they can buy.

 Example:
 For example, for 3 days the price of a stock is given as [7,10,4]. You can buy 1 stock worth 7$ on day 1, 2 stocks
 worth 4$ each on day 2, and 3 stocks worth 4$ each on day 3. If k = 100$, you can buy all the stocks (total 6) for 39$.
 For stock values [10, 7, 19] for each day, and if the customer had k = 45$ at hand, the customer can buy at most 4 stocks.
 The customer can purchase 1 stock on day 1, 2 stocks on day 2 and 1 stocks on day 3 for 10, 7*2=14 and 19 respectively.
 Hence, total amount is  and number of stocks purchased is .

 Input:
 1) arr: An array of stock values for each day.
 2) k: Initial amount of money at hand.
 Output: Maximum number of stocks to buy, under the constraint that the customer can buy at most i stocks on the i-th day.
 */

/**
 Hint:
 This is similar to Fractional Kpansack problem. We need to sort the stock values of days by increasing stock value,
 and keep the day index in order to keep the maximum number of stocks to sell that day in mind. After the (stockValue, day)
 pairs are sorted by stockValue, we will keep buying from the first stock, then the next stock, and so on, until the money
 is spent. This ensures using the smallest stock values, and maximizes the number of stock sold at the end.
 */

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;
import java.util.Comparator;

/**
 * StockDayPair class
 * Represents (stock, day) pairs.
 * Defines a comparator to sort StockDayPair objects in increasing order of stock value
 */
class StockDayPair {
    long stockValue;
    int stockDay;

    public StockDayPair(long stockValue, int stockDay) {
        this.stockValue = stockValue;
        this.stockDay = stockDay;
    }

    public long getStockValue(){
        return stockValue;
    }

    public int getStockDay(){
        return stockDay;
    }

    /**
     * Comparator to sort StockDayPair objects in increasing order of stock value
     */
    public static Comparator<StockDayPair> StockDayPairComparator = new Comparator<StockDayPair>() {
        public int compare(StockDayPair stockDayPair1, StockDayPair stockDayPair2) {
            // Sort in increasing order
            long stockValue1 = stockDayPair1.getStockValue();
            long stockValue2 = stockDayPair2.getStockValue();
            // Compare integers
            // Note: Do not use compareTo() method on integers, it returns error: "compareTo method() int cannot be dereferenced"
            // Instead, just use the normal <, >, and == operators to compare ints. Make sure to follow the contract of compareTo:
            // 1) Return an int less than 0 if stockValue1 is "less than" stockValue2,
            // 2) Return an int equal to 0 if stockValue1 is "equal to" stockValue2,
            // 3) Return an int greater than 0, if stockValue1 is "greater than" stockValue2.
            return (stockValue1 < stockValue2) ? -1 : ( (stockValue1 == stockValue2) ? 0 : 1);
        }
    };
}

public class BuyMaximumStocks {

    /**
     * Convert an array of stock values for days, into an array of (stockValue, day) pairs
     * @param stockValues
     * @return
     */
    static StockDayPair[] convertStockValuesToStockDayPair(int[] stockValues){
        StockDayPair[] stockDayPairs = new StockDayPair[stockValues.length];

        for(int i = 0; i < stockValues.length; i++){
            stockDayPairs[i] = new StockDayPair((long)stockValues[i], i+1);
        }
        return stockDayPairs;
    }

    /**
     * Return the maximum number of stocks that can be bought, given an array of stock values a, and money amount k.
     *
     * The algorithm is similar to Fractional Kpansack problem. First, we sort the stock values of days by increasing
     * stock value, and keep the day index in order to keep the maximum number of stocks to sell that day in mind.
     * After the (stockValue, day) pairs are sorted by stockValue, we will keep buying from the first stock, then the
     * next stock, and so on, until the money is spent. This ensures using the smallest stock values, and maximizes
     * the number of stock sold at the end.
     *
     * @param k: amount of money at hand
     * @param a: array of stock values for each day
     * @return
     */
    static long buyMaximumProducts(long k, int[] a) {
        long stockCount = 0;

        // Generate StockDayPair objects
        StockDayPair[] stockDayPairs = convertStockValuesToStockDayPair(a);
        // Sort StockDayPair objects by stock value in increasing order (in-place)
        Arrays.sort(stockDayPairs, StockDayPair.StockDayPairComparator);

        // Go through stockDayPair objects sorted in increasing of stockValue,
        // Buy as much as you can from the stock with smallest stock value, then move to next stock, and so on.
        for(int i = 0; i < stockDayPairs.length; i++){
            int maxStockCount = stockDayPairs[i].getStockDay();
            long stockValue = stockDayPairs[i].getStockValue();
            if(maxStockCount * stockValue <= k){
                k -= maxStockCount * stockValue;
                stockCount += (long)maxStockCount;
            } else {
                long stockCountThatFits = k / stockValue;
                k -= stockCountThatFits * stockValue;
                stockCount += stockCountThatFits;
            }
        }
        return stockCount;
    }

    public static void main(String[] args) {
        // Test 1
        // (1 * 10$) + (2 * 7$) + (1 * 19$) = 43$  => 4 stocks
        int[] arr = {10, 7, 19};
        long k = 45;
        long result = buyMaximumProducts(k, arr);
        System.out.println(result);

        // Test 2
        // (3 * 4$) + (1 * 7$) + (2 * 10$) = 39$  => 6 stocks
        int[] arr2 = {7, 10, 4};
        k = 100;
        result = buyMaximumProducts(k, arr2);
        System.out.println(result);
    }
}
