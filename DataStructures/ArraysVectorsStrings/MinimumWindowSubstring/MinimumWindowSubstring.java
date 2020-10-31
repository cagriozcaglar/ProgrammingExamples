/**
 Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

 For example,
 S = "ADOBECODEBANC"
 T = "ABC"

 Minimum window is "BANC".

 Note:
 If there is no such window in S that covers all characters in T, return the empty string "".
 If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.
 */

// Example solution: http://massivealgorithms.blogspot.com/2014/07/finding-minimum-window-in-s-which.html
// Example solution: https://medium.com/leetcode-patterns/leetcode-pattern-2-sliding-windows-for-strings-e19af105316b
// Example solution: https://www.programcreek.com/2014/05/leetcode-minimum-window-substring-java/
// Example Solution: https://github.com/awangdev/LintCode/blob/master/Java/Minimum%20Window%20Substring.java

import java.util.*;

public class MinimumWindowSubstring {
    /**
     * Sliding window method:
     * 1) Create a HashMap characterMap, which has the counts of each character in small string T.
     * 2) Iterate over the large string s:
     *   - If current character exists in characterMap of t, decrement the count of character by 1, increment the counter
     *     of used characters in t by 1.
     *     - While all characters in t are used:
     *       - Slide the window from the begin index.
     *         - If s.charAt(i) exists in t, increment counter in characterMap, decrement counter showing number of characters used from t.
     *         - Every time at this step, check if the current length of string with count == t.length() is smaller than minimum length so far.
     *
     * Time complexity: O(n), where n is the side of larger string s.
     *
     * @param s
     * @param t
     * @return
     */
    public String minWindow(String s, String t) {
        // Handle edge cases, return empty string
        if(s==null || s.length() < t.length() || s.length() == 0) {
            return "";
        }

        // Create character map of string t
        HashMap<Character, Integer> characterMap = new HashMap<Character, Integer>();
        for(char c: t.toCharArray()) {
            characterMap.put(c, characterMap.getOrDefault(c,0)+1);
        }

        // Set pointers
        int begin = 0;                      // Beginning pointer of substring
        int end = 0;                        // End pointer of substring
        int minBegin = 0;                   // Beginning index of minimum-length substring
        int minLength = s.length()+1;       // Running minimum-length of substring, set to max value possible, s.length()+1
        int count = 0;   // Number of characters in t that are used

        // Iterate over the string
        for(; end < s.length(); end++){
            char currentChar = s.charAt(end);
            // If current character exists in characterMap
            if(characterMap.containsKey(currentChar)) {
                characterMap.put(currentChar, characterMap.get(currentChar)-1); // Decrement count for currentChar in characterMap
                // If currentChar count is >=0 in characterMap, increment counter
                if(characterMap.get(currentChar) >= 0){
                    count++;
                }

                // When we find a substring with all chars in t, reduce the length of this substring
                // In this while loop, we only move the begin pointer
                while(count == t.length()){
                    // If currentLength is smaller, update minLength minBegin pointer
                    int currentLength = end-begin+1;
                    if(currentLength < minLength) {
                        minBegin = begin;
                        minLength = currentLength;
                    }
                    // If the character at the beginning index exists in characterMap
                    char beginChar = s.charAt(begin);
                    if(characterMap.containsKey(beginChar)) {
                        // Increment the count in characterMap
                        characterMap.put(beginChar, characterMap.get(beginChar)+1);
                        // If number of occurences of beginChar is > 0, decrement count
                        if(characterMap.get(beginChar) > 0) {
                            count--;
                        }
                    }
                    begin++;
                }
            } // end of if(characterMap.containsKey(currentChar))
        } // end of for

        return (minLength > s.length()) ? "" : s.substring(minBegin, minBegin+minLength);
    }

    /**
     * Run all test cases
     * @param s
     * @param t
     */
    public static void runTestCases(String s, String t) {
        System.out.println("Minimum window String in \"" + s + "\" which contains all characters of \"" + t + "\" is: \"" +
                            minWindow(s,t) + "\".");
    }

    public static void main(String[] args){
        // Test 1
        String s = "ADOBECODEBANC";
        String t = "ABC";
        runTestCases(s,t);

        // Test 2
        s = "ABCDEFNANA";
        t = "FAN";
        runTestCases(s,t);

        // Test 3: t is empty
        s = "ADOBECODEBANC";
        t = "";
        runTestCases(s,t);

        // Test 4: s is empty
        s = "";
        t = "ABC";
        runTestCases(s,t);
    }
}