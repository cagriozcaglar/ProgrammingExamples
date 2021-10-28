/**
 Given a string, you need to reverse the order of characters in each word within a sentence while still preserving
 whitespace and initial word order.

 Example 1:
 Input: "Let's take LeetCode contest"
 Output: "s'teL ekat edoCteeL tsetnoc"

 Note: In the string, each word is separated by single space and there will not be any extra space in the string.
 */

public class ReverseWordsIII {

    /**
     *
     * @param s
     * @return
     */
    public static String reverseWordsIII(String s) {
        // Error check: Check for null. Otherwise it returns NullPointerException
        if(s == null) {
            return null;
        }

        // Split the words in the original string
        String[] words = s.split(" ");

        // NOTE: Use StringBuilder to construct the new String. If you use String, it will take O(n^2) time, where n is
        // the number of characters in the original string. With StringBuilder, it takes O(n) time to construct new string.
        StringBuilder sb = new StringBuilder();

        // For each word, reverse the word using StringBuilder.reverse() method, and append it to final StringBuilder
        for(String word : words) {
            StringBuilder wordBuilder = new StringBuilder(word);
            wordBuilder.reverse(); // NOTE: StringBuilder has a .reverse() method, String does not.
            sb.append(wordBuilder + " ");  // Added " " to make sure the spaces between words are kept as well
        }
        // Why trim?: Because there will be spaces on the left / right borders of the final string.
        // E.g. the word " you" will be reversed into "uoy ", and the empty space at the end has to be removed.
        return sb.toString().trim();
    }

    /**
     * Method 2
     * @param s
     * @return
     */
    public static String reverseWordsIII2(String s) {
        // Error check: Check for null. Otherwise it returns NullPointerException
        if(s == null) {
            return null;
        }

        char[] characters = s.toCharArray();
        for (int i = 0; i < characters.length; i++) {
            if (characters[i] != ' ') {   // when i is a non-space
                int j = i;
                while (j + 1 < characters.length && characters[j + 1] != ' ') { j++; } // move j to the end of the word
                reverse(characters, i, j);
                i = j;
            }
        }
        return new String(characters);
    }

    /**
     * Helper function: Reverse characters between indices [i,j]
     * @param characters
     * @param i
     * @param j
     */
    public static void reverse(char[] characters, int i, int j){
        for(; i < j; i++, j--) {
            char tmp = characters[i];
            characters[i] = characters[j];
            characters[j] = tmp;
        }
    }

    public static void runTestCase(String s){
        System.out.println("Reverse the order of characters in each word within a sentence in \"" + s + "\": \"" + reverseWordsIII(s) + "\"");
        System.out.println("Reverse the order of characters in each word within a sentence in \"" + s + "\": \"" + reverseWordsIII2(s) + "\"");
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1: Normal case, one space between words
        String s = "Let's take LeetCode contest";
        runTestCase(s);

        // Note: Sentences with more than one space between words, or sentences with trailing spaces at the beginning and
        // end are not valid test cases, due to the note at the end of the question saying "In the string, each word is
        // separated by single space and there will not be any extra space in the string."

        // Test 2: Empty string
        s = "";
        runTestCase(s);

        // Test 3: Null string
        s = null;
        runTestCase(s);
    }
}