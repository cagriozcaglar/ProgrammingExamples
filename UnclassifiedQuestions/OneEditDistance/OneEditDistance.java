/**
 Given two strings S and T, determine if they are both one edit distance apart.
 An edit between two strings is one of the following changes.
 1) Add a character
 2) Delete a character
 3) Change a character
 Given two strings s1 and s2, find if s1 can be converted to s2 with exactly one edit.

 Expected time complexity is O(m+n) where m and n are lengths of two strings.
 */

public class OneEditDistance {
    /**
     *
     * Solution is similar to this one: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/161.%20One%20Edit%20Distance.java
     *
     * Time complexity: O(m+n) (In the worst case, we iterate over all characters of both strings)
     *
     * @param s
     * @param t
     * @return
     */
    public static boolean oneEditDistance(String s, String t) {
        // Error checks: We have to do null checks before getting lengths of strings.
        if(s == null || t == null) {
            return false;
        }

        // String lengths
        int sLength = s.length();
        int tLength = t.length();

        // Error checks based on string lengths
        if(Math.abs(sLength - tLength) > 1) {
            return false;
        }

        // Minimum of the string lengths
        int len = Math.min(sLength, tLength);

        // Iterate over the strings at the same time
        for(int i = 0; i < len; i++) {
            // Character mismatch
            if(s.charAt(i) != t.charAt(i)) {
                // Case 1: String lengths are equal. This is REPLACE operation. Rest of the strings should match.
                if(sLength == tLength) {
                    return s.substring(i+1).equals(t.substring(i+1));
                } // Case 2: First string is shorter than second string. This is ADD from s to t. Check if s[i:] and t[i+1:] match.
                else if(sLength < tLength) {
                    return s.substring(i).equals(t.substring(i+1));
                } // Case 3: First string is longer than second string. This is DELETE from s to t. Check if s[i+1:] and t[i:] match.
                else { // That is, if sLength > tLength
                    return s.substring(i+1).equals(t.substring(i));
                }
            }
        }

        // If we reached here, there was no character mismatch in s and t up to position len. However, the strings can be
        // equal, meaning they are not one-edit-distance away from each other. Therefore, return true if the length of s
        // and t differs by 1, which means that there was a deletion or addition in the last character in the strings.
        // Note: There can not be replacements, because characters of s and t matched up to this position len.
        // Note: returning true here is not correct, as it is explained above.
        return Math.abs(sLength - tLength) == 1; // Cannot be equal to 0, because that means the strings are equal.
    }

    /**
     *
     * @param s
     * @param t
     */
    public static void runTestCase(String s, String t) {
        System.out.println("Are \"" + s + "\" and \"" + t + "\" one-edit distance away?: " + oneEditDistance(s,t));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1: Replace
        String s = "best";
        String t = "beet";
        runTestCase(s,t);

        // Test 2: Add
        s = "best";
        t = "beset";
        runTestCase(s,t);

        // Test 3: Delete
        s = "best";
        t = "bst";
        runTestCase(s,t);

        // Test 4: Replace 2 characters. False
        s = "best";
        t = "base";
        runTestCase(s,t);

        // Test 5: Add 2 characters. False
        s = "best";
        t = "betest";
        runTestCase(s,t);

        // Test 6: Delete 2 characters. False
        s = "best";
        t = "be";
        runTestCase(s,t);

        // Test 7: Corner case: One of the strings is empty string: True
        s = "b";
        t = "";
        runTestCase(s,t);

        // Test 8: Corner case: One of the strings is empty string, and length difference is more than 1: False
        s = "ba";
        t = "";
        runTestCase(s,t);

        // Test 9: Corner case: One of the strings is null (in this case both). False
        s = null;
        t = null;
        runTestCase(s,t);
    }
}