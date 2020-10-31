/**
 Reverse a String
 */

public class ReverseString {

    /**
     * Key: Use StringBuffer instead of String to generate reverse String
     * @param s
     * @return
     */
    public static String reverseString1(String s) {
        // Error check: null string. If not used, this returns NullPointerException
        if(s == null) {
            return null;
        }
        char chars[] = s.toCharArray();
        // Note 1: [String vs. (StringBuffer | StringBuilder)]: We will not use String to build the reverse String,
        // because it takes O(n^2) to generate the new String. Because appending to a String creates a new object from
        // scratch. Run time is O(1+2+3+...+n) = O(n^2).
        // Note 2: [StringBuffer vs. StringBuilder]: StringBuffer allows synchronization, StringBuilder does not allow
        // synchronization
        // In other words, StringBuffer allows multi-threading, but slower. StringBuilder does not
        // allow multi-threading but it is faster.
        StringBuffer reverse = new StringBuffer();
        for(int i = chars.length - 1; i >= 0; i--) {
            reverse.append(chars[i]);
        }
        // Return String, not StringBuffer
        return reverse.toString();
    }

    /**
     * Reverse string by swapping characters
     * @param s
     * @return
     */
    public static String reverseString2(String s) {
        // Error check: null string. If not used, this returns NullPointerException
        if(s == null) {
            return null;
        }
        char chars[] = s.toCharArray();
        int i = 0;
        int j = s.length()-1;
        while(i < j) {
            // Swap chars[i] and chars[j]
            char temp = chars[i];
            chars[i] = chars[j];
            chars[j] = temp;
            // Move pointers to next / previous characters
            i++;
            j--;
        }
        // Return reverse String composed of characters in reverse order
        return new String(chars);
    }

    public static void runTestCase(String s) {
        System.out.println("Reverse of String \"" + s + "\" using StringBuffer is \"" + reverseString1(s) + "\"");
        System.out.println("Reverse of String \"" + s + "\" using in-place character swap is \"" + reverseString2(s) + "\"");
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1: Try string with odd length
        String s = "abcda";
        runTestCase(s);

        // Test 2: Try string with even length
        s = "abcdef";
        runTestCase(s);

        // Test 3: Try empty string
        s = "";
        runTestCase(s);

        // Test 4: Try null string
        s = null;
        runTestCase(s);
    }
}