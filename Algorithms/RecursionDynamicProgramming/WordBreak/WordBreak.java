/**
 Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be
 segmented into a space-separated sequence of one or more dictionary words.

 Note:
 The same word in the dictionary may be reused multiple times in the segmentation.
 You may assume the dictionary does not contain duplicate words.

 Example 1:
 Input: s = "leetcode", wordDict = ["leet", "code"]
 Output: true
 Explanation: Return true because "leetcode" can be segmented as "leet code".

 Example 2:
 Input: s = "applepenapple", wordDict = ["apple", "pen"]
 Output: true
 Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
 Note that you are allowed to reuse a dictionary word.

 Example 3:
 Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
 Output: false
*/

// TODO: 1) Add recursive solution, 2) Add tests

import java.util.*;

public class WordBreak {
    // ProgramCreek solution: https://www.programcreek.com/2012/12/leetcode-solution-word-break/
    // Retiring problem: http://thenoisychannel.com/2011/08/08/retiring-a-great-interview-problem
    // Nice solution: https://leetcode.com/problems/word-break/discuss/132611/Simple-DP-solution-Java
    public boolean wordBreak(String s, List<String> wordDict) {
        HashSet<String> wordSet = new HashSet<String>();
        for(String word : wordDict) {
            wordSet.add(word);
        }
        return wordBreak(s, wordSet);
    }

    /**
     *
     * @param s
     * @param wordSet
     * @return
     */
    public boolean wordBreak(String s, HashSet<String> wordSet) {
        // Memoization:
        // Define an array t[] such that t[i]==true => 0-(i-1) can be segmented using dictionary
        // Initial state t[0] == true
        boolean[] breakable = new boolean[s.length()+1]; // There are n+1 break points, not n
        breakable[0] = true;

        for(int i = 1; i <= s.length(); i++) {
            // Initialize to false
            breakable[i] = false;
            // Foreach each position 0 <= j < i, check if breakable[i] becomes true by splitting at j
            for(int j = 0; j < i; j++) {
                // If s[0..j-1] is breakable and s[j..i-1] is in wordSet, s[0..i] is breakable, break
                if(breakable[j] && wordSet.contains(s.substring(j,i))) {
                    breakable[i] = true;
                    break;
                }
            }
        }

        // Return breakable for the whole string s[0..n-1]
        return breakable[s.length()];
    }
}