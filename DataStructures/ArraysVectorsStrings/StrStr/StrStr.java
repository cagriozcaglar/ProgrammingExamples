/**
 Implement strStr().
 Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

 Example 1:
 Input: haystack = "hello", needle = "ll"
 Output: 2

 Example 2:
 Input: haystack = "aaaaa", needle = "bba"
 Output: -1

 Clarification:
 What should we return when needle is an empty string? This is a great question to ask during an interview.
 For the purpose of this problem, we will return 0 when needle is an empty string.
 This is consistent to C's strstr() and Java's indexOf().
 */

public class StrStr {
    /**
     * Check haystack subtring to see if they match the needle.
     * Careful: indices, off-by-one errors in String indices.
     * Note: When needle is empty string, as suggested in problem description, this method returns 0.
     *
     * Time complexity: O(haystack.length - needle.length) ~= O(haystack.length)
     *
     * @param haystack
     * @param needle
     * @return
     */
    public static int strStr(String haystack, String needle) {
        // Error cases: NOTE: Use .equals() method for String equality, otherwise this error check fails
        if(needle.equals("") || haystack == null || needle == null)  {
            return 0;
        }

        // NOTE: Careful about the end index, and careful about off-by-one errors.
        int lastIndex = haystack.length() - needle.length();

        // Iterate over the string, check if haystack substring matches needle
        for(int index = 0; index <= lastIndex; index++) {
            String haystackSubstring = haystack.substring(index, index + needle.length());
            // If there is a match, this is the first, return this first index.
            if(haystackSubstring.equals(needle)){
                return index;
            }
        }
        // Needle not found: return -1
        return -1;
    }

    public static void runTestCase(String haystack, String needle) {
        System.out.println("The first index in \"" + haystack + "\" that matches \"" + needle + "\" is : " +
                           strStr(haystack, needle));
    }

    public static void main(String[] args) {
        // Test 1
        String haystack = "hello";
        String needle = "ll";
        runTestCase(haystack, needle);

        // Test 2
        haystack = "abcdef";
        needle = "cd";
        runTestCase(haystack, needle);

        // Test 3: Needle does not exist
        haystack = "abcdef";
        needle = "klm";
        runTestCase(haystack, needle);

        // Test 4: Haystack and needle are one letter strings that match
        haystack = "a";
        needle = "a";
        runTestCase(haystack, needle);

        // Test 5: Haystack is empty string
        haystack = "";
        needle = "abc";
        runTestCase(haystack, needle);

        // Test 6: Needle is empty string
        haystack = "abc";
        needle = "";
        runTestCase(haystack, needle);

        // Test 7: Both haystack and needle are empty strings
        haystack = "";
        needle = "";
        runTestCase(haystack, needle);
    }
}