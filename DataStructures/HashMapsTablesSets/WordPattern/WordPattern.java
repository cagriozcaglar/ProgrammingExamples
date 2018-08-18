/**
 Given a pattern and a string str, find if str follows the same pattern.
 Here follow means a full match, such that there is a **bijection** between a letter in pattern and a non-empty word in str.

 Examples:
 1. pattern = "abba", str = "dog cat cat dog" should return true.
 2. pattern = "abba", str = "dog cat cat fish" should return false.
 3. pattern = "aaaa", str = "dog cat cat dog" should return false.
 4. pattern = "abba", str = "dog dog dog dog" should return false.

 Notes:
 You may assume pattern contains only lowercase letters, and str contains lowercase letters separated by a single space.
 */

import java.util.*;

public class WordPattern {
    /**
     * Given a pattern (e.g. "abba") and a string (e.g. "dog cat cat dog"), check if string follows the same pattern.
     * That is, there is a bijection (1:1 mapping) between a letter in pattern and non-empty word in string.
     * Note: the relationship is 1:1, which means pattern1 <-> word1, pattern2 <-> word2. Different patterns have to map
     * to different words, and different words have to map to different patterns.
     * @param pattern
     * @param str
     * @return
     */
    public static boolean wordPattern(String pattern, String str) {
        String[] words = str.split(" ");
        char[] patterns = pattern.toCharArray();

        // If lengths are not equal, return false
        if(patterns.length != words.length){
            return false;
        }

        // Pattern to word map
        HashMap<Character, String> patternToWordMap = new HashMap<Character, String>();
        // Word to pattern map
        HashMap<String, Character> wordToPatternMap = new HashMap<String, Character>();

        // Iterate over patterns and words in parallel. Check if there is a change in pattern <-> word map and
        // word <-> pattern map
        for(int i = 0; i < patterns.length; i++){
            // If pattern is already visited
            if(patternToWordMap.containsKey(patterns[i])){
                // If pattern -> word map needs to be updated, return false (same pattern maps to two different words)
                if( ! patternToWordMap.get(patterns[i]).equals(words[i]) ){
                    return false;
                }
            } else { // If this is the first observation of the pattern
                // If word exists in word -> pattern map (but pattern does not exist in pattern -> word map)
                // This means that the word is already matched with another pattern. One word, two patterns, not 1:1.
                if(wordToPatternMap.containsKey(words[i])){
                    return false;
                }
                // Otherwise, add pattern->word to patternToWordMap, word->pattern to wordToPatternMap
                patternToWordMap.put(patterns[i], words[i]);
                wordToPatternMap.put(words[i], patterns[i]);
            }
        }

        return true;
    }

    public static void main(String[] args){
        // Test 1: True
        String pattern = "abba";
        String word = "dog cat cat dog";
        System.out.println(wordPattern(pattern, word));

        // Test 2: False
        pattern = "abba";
        word = "dog cat cat fish";
        System.out.println(wordPattern(pattern, word));

        // Test 3: False
        pattern = "aaaa";
        word = "dog cat cat dog";
        System.out.println(wordPattern(pattern, word));

        // Test 4: False (Careful: a->dog, b->dog, but two letters map to the same word, hence not a bijection)
        pattern = "abba";
        word = "dog dog dog dog";
        System.out.println(wordPattern(pattern, word));
    }
}
