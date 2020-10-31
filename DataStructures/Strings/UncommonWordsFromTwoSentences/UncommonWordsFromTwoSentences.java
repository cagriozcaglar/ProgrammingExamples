/**
 We are given two sentences A and B. (A sentence is a string of space separated words. Each word consists only of lowercase
 letters.). A word is uncommon **if it appears exactly once in one of the sentences, and does not appear in the other sentence.**
 Return a list of all uncommon words. You may return the list in any order.

 Example 1:
 Input: A = "this apple is sweet", B = "this apple is sour"
 Output: ["sweet","sour"]

 Example 2:
 Input: A = "apple apple", B = "banana"
 Output: ["banana"]

 Note:
 0 <= A.length <= 200
 0 <= B.length <= 200
 A and B both contain only spaces and lowercase letters.
 */

import java.util.*;

public class UncommonWordsFromTwoSentences {

    /**
     * 1) Concatenate strings A and B, then split, creating an array of all words: (A + " " + B).split(" ")
     * 2) Keep a HashMap from each word to its count. If the count is 1, word is uncommon.
     * @param A
     * @param B
     * @return
     */
    public static String[] uncommonWordsFromTwoSentencesShort(String A, String B) {
        String[] allWords = (A + " " + B).trim().split(" ");

        // Create word counter HashMap
        HashMap<String, Integer> wordCounter = new HashMap<String, Integer>();
        for(String word : allWords) {
            wordCounter.put(word, wordCounter.getOrDefault(word, 0) + 1);
        }

        // Create uncommon words list (variable size), then convert to array
        ArrayList<String> uncommonWordsList = new ArrayList<String>();
        for(String word : wordCounter.keySet()) {
            if(wordCounter.get(word) == 1) {
                uncommonWordsList.add(word);
            }
        }

        // Return uncommon words array
        return uncommonWordsList.toArray(new String[uncommonWordsList.size()]);
    }

    /**
     * Solution with two HashSets, takes longer, time and space-wise
     * @param A
     * @param B
     * @return
     */
    public static String[] uncommonWordsFromTwoSentencesLong(String A, String B) {
        String[] wordsA = A.trim().split(" ");
        String[] wordsB = B.trim().split(" ");

        // Combine word arrays into one word list, without losing duplication
        List<String> combinedWordList = new ArrayList<String>(Arrays.asList(wordsA));
        combinedWordList.addAll(Arrays.asList(wordsB));

        // Uncommon words HashSet: When the word appears the first time, add it.
        // When it appears the second time, remove it.
        HashSet<String> uncommonWords = new HashSet<String>();
        HashSet<String> commonWords = new HashSet<String>();

        // Iterate over combinedWordList
        for(String theWord : combinedWordList) {
            if( !uncommonWords.contains(theWord) && !commonWords.contains(theWord) ) {
                uncommonWords.add(theWord);
            } else {
                uncommonWords.remove(theWord);
                commonWords.add(theWord);
            }
        }

        // Return uncommonWords HashSet as a String array
        return uncommonWords.toArray(new String[uncommonWords.size()]);
    }

    /**
     * Run a test case with all solutions
     * @param A
     * @param B
     */
    public static void runTestCase(String A, String B) {
        System.out.println("Uncommon words from \"" + A + "\" and \"" + B + "\" using short method are: " +
                            Arrays.toString(uncommonWordsFromTwoSentencesShort(A,B)));
        System.out.println("Uncommon words from \"" + A + "\" and \"" + B + "\" using long method are: " +
                            Arrays.toString(uncommonWordsFromTwoSentencesLong(A,B)));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1: Regular case
        String A = "this apple is sweet";
        String B = "this apple is sour";
        runTestCase(A,B);

        // Test 2: A word is repeated in the same sentence
        A = "apple apple";
        B = "banana";
        runTestCase(A,B);

    }
}