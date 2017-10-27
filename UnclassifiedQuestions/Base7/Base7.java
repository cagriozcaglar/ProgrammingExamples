/**
 Given an integer, return its base 7 string representation.

 Example 1:
 Input: 100
 Output: "202"

 Example 2:
 Input: -7
 Output: "-10"

 Note: The input will be in range of [-1e7, 1e7].
 */

public class Base7 {
    /**
     * Solution 1: Iterative solution, we keep dividing the number by base, until the quotient is less than the base.
     * We use StringBuilder and append the remainders to the beginning of the string first, then the final quotient,
     * and then the sign of the number if it is negative.
     * Time Complexity: O(log7(num))
     *
     * Example calculation: number = 100.
     * Step 1:
     *  100 / 7 = 14 (quotient)
     *  100 % 7 = 2 (remainder)
     *  result = "2" (append remainder at the beginning of the string)
     * Step 2:
     *  14 / 7 = 2 (quotient)
     *  14 % 7 = 0 (remainder)
     *  result = "02" (append remainder at the beginning of the string)
     * Step 3:
     *  Since 2 < 7 (quotient < base), append quotient to the beginning of result:
     *  result = "202"
     * @param num
     * @return
     */
    public static String convertToBase7_1(int num) {
        int base = 7;
        StringBuilder result = new StringBuilder("");
        int number = Math.abs(num);
        int remainder, value;

        // Add remainder values by appending them to the beginning of the string
        while(number >= base){
            remainder = number % base;
            number = number / base;
            result.insert(0, remainder);
        }

        // Add the final quotient by appending it to the beginning of the string
        if(number < base){
            result.insert(0, number);
        }

        // Add the sign of the number by appending it to the beginning of the string
        if(num < 0){
            result.insert(0, "-");
        }

        return result.toString();
    }

    /**
     * Solution 2: Recursive solution.
     * Given the number, we keep dividing the number by the base, until quotient is 0.
     *  1) Base case: Quotient is 0 at the end of the continuous division, return the remainder value.
     *  2) Recursive case: If quotient > 0, append the current remainder at the front of the string, call the method on
     *     quotient, and append its result to the front of the string.
     * @param num
     * @return
     */
    public static String convertToBase7_2(int num){
        int base = 7;
        int quotient = num / base;
        int remainder = num % base;

        // Base case: Quotient is 0 at the end of the continuous division, return the remainder value.
        // Note: When quotient < base, this method continues to divide the quotient by base, until quotient == 0. In
        // this case, the last quotient that is < base, becomes the last remainder, which is returned at this if condition
        if(quotient == 0){
            return Integer.toString(remainder);
        }
        // Recursive case: If quotient > 0, append the current remainder at the front of the string, call the method
        // on quotient, and append its result to the front of the string.
        // Note: Result of recursive method call is returned first, to be appended at the beginning of the string. Then,
        // the current remainder is appended to the string.
        else {
            return convertToBase7_2(quotient) + Integer.toString(remainder);
        }
    }

    /**
     * Solution 3: Use Integer.toString(int i, int radix). One-liner.
     * More: https://docs.oracle.com/javase/7/docs/api/java/lang/Integer.html#toString(int,%20int)
     * @param num
     * @return
     */
    public static String convertToBase7_3(int num){
        return Integer.toString(num, 7);
    }

    /**
     * Run a test case with all solutions
     * @param number
     */
    public static void runTestCase(int number){
        System.out.println("convertToBase7_1(" + number + "): " + convertToBase7_1(number));
        System.out.println("convertToBase7_2(" + number + "): " + convertToBase7_2(number));
        System.out.println("convertToBase7_3(" + number + "): " + convertToBase7_3(number));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1: Positive number
        int number = 100;
        runTestCase(number);

        // Test 2: Negative number
        number = -7;
        runTestCase(number);

        // Test 3: Zero
        number = 0;
        runTestCase(number);

        // Test 4: Positive number in the range [1, base]
        number = 4;
        runTestCase(number);

        // Test 5: Negative number with absolute value in the range [1, base]
        number = -5;
        runTestCase(number);
    }
}