/**
 Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2 as a string.

 Note:

 The length of both num1 and num2 is < 5100.
 Both num1 and num2 contains only digits 0-9.
 Both num1 and num2 does not contain any leading zero.
 You must not use any built-in BigInteger library or convert the inputs to integer directly.
 */

/**
 * Note: The first solution that comes to mind is to iterate over strings starting from right, convert them to integer,
 * multiply with the exponent, add to the sum, and at the end convert sum to String. However, this sum can be large, larger than
 * maximum integer, which requires the use of Java.lang.BigInteger, which is not allowed for this question. Therefore, you
 * need to start from right, do addition bit-by-bit and save the results as a string, keep the carry to next bit, and continue.
 */

class AddStrings {

    /**
     * Add two numbers in the form of a string, and return the sum as a string, without using BigInteger class.
     * Start from right, do addition bit-by-bit and save the results as a string, keep the carry to next bit, and continue.
     * @param num1
     * @param num2
     * @return
     */
    public static String addStrings(String num1, String num2) {

        StringBuilder sumString = new StringBuilder("");

        int length1 = num1.length();
        int length2 = num2.length();
        int maxLength = Math.max(length1, length2);

        // Left zero-padding on the short string, so that strings are of same length
        if(length1 > length2){
            num2 = padZerosLeft(num2, length1-length2);
        } else if(length1 < length2){
            num1 = padZerosLeft(num1, length2-length1);
        }

        // Interim variables
        int integerValue1 = 0;
        int integerValue2 = 0;
        int integerValue = 0;
        int value = 0;
        int carry = 0;

        for(int i = maxLength - 1; i >= 0; i--){
            integerValue1 = Integer.parseInt( num1.substring(i, i + 1) );
            integerValue2 = Integer.parseInt( num2.substring(i, i + 1) );
            integerValue = integerValue1 + integerValue2 + carry;
            value = integerValue % 10;
            carry = integerValue / 10;
            // System.out.println("value: " + value + ", carry: " + carry);
            sumString.insert(0, Integer.toString(value));
        }

        // Add carry if any left.
        if(carry > 0) {
            sumString.insert(0, Integer.toString(carry));
        }

        // Return the sum as a string (StringBuilder -> String conversion)
        return sumString.toString();
    }

    /**
     * Pad zeros to the left of a string s for k times
     * @param s: initial string
     * @param k: number of zeros to pad on the left of the initial string
     * @return
     */
    public static String padZerosLeft(String s, int k){
        StringBuilder sb = new StringBuilder(s);
        for(int i=0; i < k; i++){
            sb.insert(0, "0");
        }
        return sb.toString();
    }

    public static void main(String[] args){
        // Test: (Expected: "6984362587")
        String num1 = "6913259244";
        String num2 =   "71103343";
        System.out.println( addStrings(num1, num2) );

        // Testing left-zero-padding
        System.out.println(padZerosLeft("abc",4));
    }
}
