/**
 A self-dividing number is a number that is divisible by every digit it contains.
 For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.
 Also, a self-dividing number is not allowed to contain the digit zero.
 Given a lower and upper number bound, output a list of every possible self dividing number, including the bounds if possible.

 Example 1:
 Input: left = 1, right = 22
 Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]

 Note:
 The boundaries of each input argument are 1 <= left <= right <= 10000.
 */

import java.util.*; // For using ArrayList / List

public class SelfDividingNumber {
    /**
     * Solution: For each number in range [left, right], check if the number is a self dividing number.
     * Note: When checking if the number is a self-dividing number, make sure to keep the original value of the number
     * (n), and use a running value of the number for getting the digits of the number.
     * @param left
     * @param right
     * @return
     */
    public static List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> numberList = new ArrayList<Integer>();
        for(int num = left; num <= right; num++){
            if(isSelfDividingNumber(num)) {
                numberList.add(num);
            }
        }
        return numberList;
    }

    /**
     * Helper for the solution. Checks if a number is a self-dividing number.
     * Note: When checking if the number is a self-dividing number, make sure to keep the original value of the number
     * (n), and use a running value of the number for getting the digits of the number.
     * @param n
     * @return
     */
    public static boolean isSelfDividingNumber(int n){
        int currentDigit = 0;
        int runner = n;  // You need the actual value of n for division
        while(runner > 0){
            currentDigit = runner % 10;  // current digit
            // If current digit is 0, or is n is not divisible by current digit, return false.
            if(currentDigit == 0 || n % currentDigit != 0){
                return false;
            }
            runner = runner/10; // Move to next digit by dividing by 10.
        }
        return true;
    }

    /**
     * Run a test case with all solutions
     * @param left
     * @param right
     */
    public static void runTestCase(int left, int right){
        // NOTE: You can print contents of a list by [listName].toString(), or even better by [listName]
        System.out.println(selfDividingNumbers(left,right));
        // System.out.println(selfDividingNumbers(left,right).toString());  // Print list with .toString() method
    }

    public static void main(String[] args){
        // Test 1
        int left = 1;
        int right = 22;
        runTestCase(left, right);

        // Test 2
        left = 456;
        right = 543;
        runTestCase(left, right);
    }
}