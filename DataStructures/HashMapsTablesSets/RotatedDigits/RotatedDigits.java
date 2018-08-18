/**
 X is a *good number* if after rotating each digit individually by 180 degrees, we get a valid number that is different from X.
 A number is *valid* if each digit remains a digit after rotation.
 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other; 6 and 9 rotate to each other,
 and the rest of the numbers do not rotate to any other number.
 Now, given a positive number N, how many numbers X from 1 to N are good?

 Example:
 Input: 10
 Output: 4
 Explanation:
 There are four good numbers in the range [1, 10] : 2, 5, 6, 9.
 Note that 1 and 10 are not good numbers, since they remain unchanged after rotating.

 Note: N  will be in range [1, 10000].
 */

import java.util.*; // For HashMap

public class RotatedDigits {
    /**
     * Return the number of "good number"s between 1 and N, both inclusive.
     * A number is a *good number* if after rotating each digit individually by 180 degrees, we get a valid number that is
     * different from the original number. A number is *valid* if each digit remains a digit after rotation.
     * 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other; 6 and 9 rotate to each other,
     * and the rest of the numbers (3,4,7) do not rotate to any other number.
     *
     * Time complexity: O(N * d), where d is the average string length of numbers between 1 and N, which is negligible.
     *
     * @param N
     * @return
     */
    public static int rotatedDigits(int N) {

        // Initialize the digit -> rotated digit map. Note that 3,4,7 do not have rotated digits.
        // NOTE: Methods for inline HashMap initialization: https://javatutorial.net/java-hashmap-inline-initialization
        /**
         -- 0,1,8 rotate to themselves
         -- 2,5 rotate to each other
         -- 6,9 rotate to each other
         -- 3,4,7: do not rotate to any other number
         */
        HashMap<Character,Character> rotationMap = new HashMap<Character, Character>() {
            {
                put('0','0');
                put('1','1');
                put('2','5');
                put('5','2');
                put('6','9');
                put('8','8');
                put('9','6');
            }
        };

        // Running count of good numbers
        int count = 0;
        // Iterate over numbers in [1,N]
        for(int number = 1; number <= N; number++){
            // Convert the number to String, for checking each digit
            String numberString = Integer.toString(number);
            // Set allRotatableChars to true, which means the string has all rotatable digits at the beginning
            boolean allRotatableChars = true;
            // Build the rotated digit string using StringBuilder, instead of a String (O(n) vs. O(n^2), where n is length.)
            StringBuilder rotatedDigitString = new StringBuilder("");
            // Test 1: Check if the number does not have 3,4,7 in it: All rotatable characters / digits
            for(int i = 0; i < numberString.length(); i++){
                Character currentChar = numberString.charAt(i);
                // If currentChar is false, not all characters are rotatable, and the number is not a good number
                if(! rotationMap.containsKey(currentChar)){
                    allRotatableChars = false;
                    break;
                } // Otherwise, append the currentChar to rotated digit string.
                else {
                    rotatedDigitString.append(rotationMap.get(currentChar));
                }
            }
            // Test 2: All rotatable characters & rotated digit string is different from original string
            if( allRotatableChars &&  // Composed of all rotatable characters
                !numberString.equals(rotatedDigitString.toString()) ){ // Original and rotated strings are different
                count++; // Good number found, increase the count
            }
        }

        return count++;
    }

    public static void runTestCase(int number){
        System.out.println( "rotatedDigits(" + Integer.toString(number) + "): " + rotatedDigits(number) );
    }

    public static void main(String[] args) {
        // Test 1
        int number = 10;
        runTestCase(number);

        // Test 2
        number = 20;
        runTestCase(number);

        // Test 3
        number = 100;
        runTestCase(number);
    }
}