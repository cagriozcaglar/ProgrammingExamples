/**
 Given a string, find *the length* of the longest substring without repeating characters.

 Examples:
 1. Given "abcabcbb", the answer is "abc", which the length is 3.
 2. Given "bbbbb", the answer is "b", with the length of 1.
 3. Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring,
    "pwke" is a subsequence and not a substring.
 */

import java.util.*;

class LongestSubstringWithoutRepeatingCharacters {
    /**
     * Solution 1: Iterative, with two pointers, O(n^2). (Can be done better, see other solutions below).
     * Iterate over the string starting at each position, and continuing with the second index until
     * you hit a repeated character. Keep a counter on current length of the string and max length of non-repeating
     * substring. Finally, return the max length of non-repeating substring.
     * Time complexity: O(n^2) (Not good time-wise, check other solutions which are better).
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring1(String s) {
        int maxLength = 0;

        for(int i = 0; i < s.length(); i++){
            // Initialize current length and current string set
            // NOTE: Using HashSet requires "import java.util.*;"
            HashSet<Character> currentStringSet = new HashSet<Character>();
            int currentLength = 0;
            for(int j = i; j < s.length(); j++){
                char val = s.charAt(j);
                if(!currentStringSet.contains(val)){
                    currentStringSet.add(val);
                    currentLength++;
                    maxLength = Math.max(maxLength, currentLength);
                } else {
                    break;
                }
            }
        }
        return maxLength;
    }

    /**
     * Solution 2: Using two pointers, iterative, in O(n) time.
     * Use a HashSet to track the longest substring without repeating characters so far, use a "fast" pointer for the end
     * of the substring and "slow" pointer for the beginning of the substring. Use the fast pointer to see if character
     * at "fast" is in the HashSet or not. Two outcomes:
     * 1) If character at "fast" is not in the HashSet, add it to the HashSet, update maxLength, increment "fast" pointer
     * 2) If character at "fast" is in the HashSet, delete the character at index "slow" from the HashSet (because that is
     * the repeating character (otherwise the iteration would not have reached thus far)), and increment "slow" pointer
     * Time Complexity: O(n)
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring2(String s){
        // Two pointers (slow, fast), and maxLength variable to hold the maximum length
        int slow = 0;
        int fast = 0;
        int maxLength = 0;

        // Non-repeating character set so far
        Set<Character> charSet = new HashSet<>();

        // Iterate over the string with fast and slow pointers
        while(fast < s.length()){
            // If character does not exist in set, add it to set, increase maxLength, proceed to next character
            if(!charSet.contains(s.charAt(fast))){
                charSet.add(s.charAt(fast));
                fast++;
                maxLength = Math.max(maxLength, charSet.size());
            } // If character exists in the set, remove the character at index "slow" from the set (because that is the
              // repeating character (otherwise the iteration would not have reached thus far)), and increment "slow" pointer
            else{
                charSet.remove(s.charAt(slow));
                slow++;
            }
        }
        return maxLength;
    }

    /**
     * Solution 3: Dynamic programming. Overkill, but works in O(n) time
     * Assume L[i] = s[m...i], denotes the longest substring without repeating characters that ends up at s[i], and we
     * keep a hashmap for every characters between m ... i, while storing <character, index> in the hashmap.
     * We know that each character will appear only once. Then to find s[i+1]:
     * 1) If s[i+1] does not appear in hashmap, we can just add s[i+1] to hash map. and L[i+1] = s[m...i+1]
     * 2) If s[i+1] exists in hashmap, and the hashmap value (the index) is k, let m = max(m, k),
     * then L[i+1] = s[m...i+1], we also need to update entry in hashmap to mark the latest occurency of s[i+1].
     * Since we scan the string for only once, and the 'm' will also move from
     * beginning to end for at most once. Overall complexity is O(n).
     * Time Complexity: O(n)
     * @param s
     * @return
     */
    public static int lengthOfLongestSubstring3(String s){
        HashMap<Character, Integer> charToIndexMap = new HashMap<Character, Integer>();
        int maxLength = 0;
        int m = 0;

        for(int i = 0; i < s.length(); i++){
            char currentChar = s.charAt(i);
            // Get max of index of current character and m.
            // Note: In the HashMap, the default value is -1, and +1 is to take care of -1 case
            m = Math.max(charToIndexMap.getOrDefault(currentChar, -1) + 1, m);
            charToIndexMap.put(currentChar, i);
            maxLength = Math.max(maxLength, (i-m+1));
        }
        return maxLength;
    }

    /**
     * Run a test case with all solutions
     * @param str
     */
    public static void runTestCase(String str){
        System.out.println("lengthOfLongestSubstring1(" + "\"" + str + "\"): " + lengthOfLongestSubstring1(str));
        System.out.println("lengthOfLongestSubstring2(" + "\"" + str + "\"): " + lengthOfLongestSubstring2(str));
        System.out.println("lengthOfLongestSubstring3(" + "\"" + str + "\"): " + lengthOfLongestSubstring3(str));
    }

    public static void main(String[] args){
        // Test 1
        String s = "abcabcbb";
        runTestCase(s);

        // Test 2
        s = "bbbbb";
        runTestCase(s);

        // Test 3
        s = "pwwkew";
        runTestCase(s);

        // Test 4
        s = "";
        runTestCase(s);
    }
}
