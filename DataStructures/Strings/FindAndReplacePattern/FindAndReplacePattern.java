/**
 You have a list of words and a pattern, and you want to know which words in words matches the pattern.
 A word matches the pattern if there exists a permutation of letters p so that after replacing every letter x in the
 pattern with p(x), we get the desired word.
 (Recall that a permutation of letters is a bijection from letters to letters: every letter maps to another letter, and
 no two letters map to the same letter.)
 Return a list of the words in words that match the given pattern.
 You may return the answer in any order.

 Example 1:
 Input: words = ["abc","deq","mee","aqq","dkd","ccc"], pattern = "abb"
 Output: ["mee","aqq"]
 Explanation: "mee" matches the pattern because there is a permutation {a -> m, b -> e, ...}.
 "ccc" does not match the pattern because {a -> c, b -> c, ...} is not a permutation,
 since a and b map to the same letter.

 Note:
 1 <= words.length <= 50
 1 <= pattern.length = words[i].length <= 20
 */

import java.util.*;

public class FindAndReplacePattern {
    /**
     *
     * @param words
     * @param pattern
     * @return
     */
    public static List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> wordsPatternMatch = new ArrayList<String>();

        // Iterate over all words, and check if word->pattern mapping is valid
        for(String word : words) {
            if(isPermutePattern(word, pattern)) {
                wordsPatternMatch.add(word);
            }
        }

        return wordsPatternMatch;
    }

    /**
     * Check if a word matches the pattern, that is, if there exists a permutation of letters p so that after replacing
     * every letter x in the pattern with p(x), we get the desired word.
     * Note: A permutation of letters is a bijection from letters to letters: every letter maps to another letter, and
     * no two letters map to the same letter. This means that we need to check both wordChar -> patternChar mappings and
     * patternChar -> wordChar mappings.
     * @param word
     * @param pattern
     * @return
     */
    public static boolean isPermutePattern(String word, String pattern) {
        if(word.length() != pattern.length()) {
            return false;
        }
        HashMap<Character, Character> wordToPatternMap = new HashMap<>();
        HashMap<Character, Character> patternToWordMap = new HashMap<>();
        char[] wordChars = word.toCharArray();
        char[] patternChars = pattern.toCharArray();

        // Iterate over characters of word and pattern in parallel
        for(int i = 0; i < wordChars.length; i++) {
            char wordChar = wordChars[i];
            char patternChar = patternChars[i];

            // Check wordChar -> patternChar mapping
            if(wordToPatternMap.containsKey(wordChar)) {
                if(wordToPatternMap.get(wordChar) != patternChar) {
                    return false;
                }
            } else {
                wordToPatternMap.put(wordChar, patternChar);
            }

            // Check patternChar -> wordChar mapping
            if(patternToWordMap.containsKey(patternChar)) {
                if(patternToWordMap.get(patternChar) != wordChar) {
                    return false;
                }
            } else {
                patternToWordMap.put(patternChar, wordChar);
            }
        }

        // If all checks passed till this point, return true
        return true;
    }

    /**
     * Run a test case with solutions
     * @param words
     * @param pattern
     */
    public static void runTestCase(String[] words, String pattern) {
        System.out.println("The words in \"" + Arrays.toString(words) + "\" that match pattern " +"\"" + pattern + "\": " +
                            findAndReplacePattern(words, pattern));
    }

    public static void main(String[] args) {
        // Test 1
        String[] words = new String[]{"abc","deq","mee","aqq","dkd","ccc"};
        String pattern = "abb";
        runTestCase(words, pattern);
    }
}
