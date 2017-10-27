/**
 Reverse digits of an integer.

 Example1: x = 123, return 321
 Example2: x = -123, return -321

 Note:
 The input is assumed to be a 32-bit signed integer. Your function should return 0 when the reversed integer overflows.
 */

/**
 Spoilers:
 Have you thought about this?
 Here are some good questions to ask before coding. Bonus points for you if you have already thought through this!
 1) If the integer's last digit is 0, what should the output be? ie, cases such as 10, 100.
 2) Did you notice that the reversed integer might overflow? Assume the input is a 32-bit integer, then the reverse of
    1000000003 overflows. How should you handle such cases? For the purpose of this problem, assume that your function
    returns 0 when the reversed integer overflows.
 */

import java.util.*; // For timer

public class ReverseInteger {

    /**
     * Solution 1: Mathematical calculation, faster.
     * Divide the number by 10, the remainder is the first digit of the reverse number, quotient is used in
     * the next iteration. When shifting to next digit, multiply the value of reverse number, as you are shifting the
     * reverse number to the left by 1 position. Finally, if the reverseNumber causes overflow at any point (by being
     * outside [Integer.MIN_VALUE, Integer.MAX_VALUE]) range, return 0. Otherwise, after iterating over all digits, return
     * the reverse number.
     * @param x
     * @return
     */
    public static int reverse1(int x) {
        int quotient, remainder;
        long reverseNumber = 0;

        while(x != 0) {
            quotient = x / 10;
            remainder = x % 10;
            reverseNumber = reverseNumber * 10 + remainder;
            x = quotient;
            // Overflow check
            if(reverseNumber > Integer.MAX_VALUE || reverseNumber < Integer.MIN_VALUE){
                return 0;
            }
        }
        return (int) reverseNumber;
    }

    /**
     * Solution 2: Naive reverse string approach, slower.
     * Convert integer to long first, to prevent overflow. Extract absolute value and sign of the long number.
     * Convert absolute value to String, reverse the String (using StringBuilder.reverse() method), append the sign string
     * to the beginning of the reverse string if sign exists. Then convert reverse string to long, check if the long value
     * causes overflow at any point (by being outside [Integer.MIN_VALUE, Integer.MAX_VALUE]) range, in which case return 0.
     * Otherwise, convert reverse string to integer and return the value.
     * @param x
     * @return
     */
    public static int reverse2(int x) {
        // Extract absolute value and sign of the integer
        // NOTE: If you use integer x and use Math.abs(x), this returns an error for the Integer.MIN_VALUE: -2147483648.
        // Because, Integer.MAX_VALUE = +2147483647, and reverse of (Integer.MIN_VALUE = -2147483648) causes overflow,
        // if we pass integer x as the argument to Math.abs(x). But, if we convert it to long, and then call Math.abs(x),
        // the overflow error is prevented this way.
        // See more details here: https://stackoverflow.com/questions/5444611/math-abs-returns-wrong-value-for-integer-min-value
        // Also check Java docs for Math.abs(): https://docs.oracle.com/javase/6/docs/api/java/lang/Math.html#abs%28int%29
        long absoluteValue = Math.abs((long)x);
        String signString = (x >= 0) ? "" : "-" ;

        // Get reverse of the absolute value,
        StringBuilder reverseValueStringBuilder = new StringBuilder(Long.toString(absoluteValue));
        reverseValueStringBuilder.reverse();
        reverseValueStringBuilder.insert(0, signString);
        String reverseValueString = reverseValueStringBuilder.toString();

        // Convert reverse of absolute value to long (due to possible overflow)
        long reverseValueLong = Long.parseLong(reverseValueString);

        // Return the reverse integer, while handling the overflow case
        return (reverseValueLong > Integer.MAX_VALUE || reverseValueLong < Integer.MIN_VALUE) ? 0 :  Integer.parseInt(reverseValueString);
    }

    /**
     * Run a test case with all solutions, with function time profiling
     * @param number
     */
    public static void runTestCase(int number){
        // Timers for profiling
        long startTime;
        long endTime;

        // Solution 1
        startTime = System.nanoTime();
        int reverseNumber1 = reverse1(number);
        endTime =  System.nanoTime();
        System.out.println("reverse1(" + number + "): " + reverseNumber1 + ". Time: " + (endTime - startTime) + " ns");

        // Solution 2
        startTime = System.nanoTime();
        int reverseNumber2 = reverse2(number);
        endTime =  System.nanoTime();
        System.out.println("reverse2(" + number + "): " + reverseNumber2 + ". Time: " + (endTime - startTime) + " ns");

        System.out.println();
    }

    public static void main(String[] args){
        // Test case 1: Positive integer
        int number = 101;
        runTestCase(number);

        // Test case 2: Negative integer
        number = -178;
        runTestCase(number);

        // Test case 3: Zero
        number = 0;
        runTestCase(number);

        // Test case 4: Maximum integer
        number = 2147483647;
        runTestCase(number);

        // Test case 5: Maximum integer
        number = -2147483648;
        runTestCase(number);

        // Test case 6: Valid positive integer within range, but the reverse overflows
        number = 1000000007;
        runTestCase(number);

        // Test case 7: Valid negative integer within range, but the reverse overflows
        number = -1000000007;
        runTestCase(number);

        /** Output: Shows that solution 1 is faster than solution 2.

         reverse1(101): 101. Time: 2018 ns
         reverse2(101): 101. Time: 153899 ns

         reverse1(-178): -871. Time: 801 ns
         reverse2(-178): -871. Time: 11886 ns

         reverse1(0): 0. Time: 277 ns
         reverse2(0): 0. Time: 7016 ns

         reverse1(2147483647): 0. Time: 1038 ns
         reverse2(2147483647): 0. Time: 11605 ns

         reverse1(-2147483648): 0. Time: 1013 ns
         reverse2(-2147483648): 0. Time: 15886 ns

         reverse1(1000000007): 0. Time: 1058 ns
         reverse2(1000000007): 0. Time: 16647 ns

         reverse1(-1000000007): 0. Time: 942 ns
         reverse2(-1000000007): 0. Time: 19629 ns
         */

    }
}