/**
 Given an input string, reverse the string word by word.

 Example:
 Input: "the sky is blue",
 Output: "blue is sky the".

 Note:
 A word is defined as a sequence of non-space characters.
 Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
 You need to reduce multiple spaces between two words to a single space in the reversed string.

 Follow up: For C programmers, try to solve it in-place in O(1) space.
 */

public class ReverseWordsInString {
    public static String reverseWordsInString(String s) {
        // Trim string from ends, Split string by one space character
        // Example splits:
        // "b a" => ["b", "a"]
        // "b  a" => ["b", " ", "a"]
        // "b   a" => ["b", " ", " ", "a"]
        String[] words = s.trim().split(" +");       // Split by one or more spaces
        // String[] words = s.trim().split("\\s+");  // Split by all whitespace characters: https://stackoverflow.com/questions/225337/how-do-i-split-a-string-with-any-whitespace-chars-as-delimiters?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
        // String[] words = s.trim().split(" ");     // Split by one space character
        StringBuilder reversedWords = new StringBuilder();

        // Iterate over words array in reverse order (Careful: start from words.length-1, not words.length)
        for(int i = words.length-1; i >= 0; i--) {
            // Note: If the original string is split by " +" (one or more spaces), you can remove the if condition and
            // only run the statement inside it.
            if(!words[i].equals("")){  // Careful here, not " ", it is "", empty string. Because trim() will return empty string.
                reversedWords.append(words[i].trim()).append(" "); // Careful: Do not use String addition / concatenation for " ", that causes time complexity increase
            }
        }
        // Careful here:
        // Case 1: If string is empty, return the empty string.
        // Case 2: If string is not empty, there is an extra space at the end, remove the space, return
        return reversedWords.length()==0 ? "" : reversedWords.substring(0, reversedWords.length()-1);
    }

    public static void runTestCase(String text) {
        System.out.println("Reverse words in string \"" + text + "\": \"" + reverseWordsInString(text) + "\"");
    }

    public static void main(String[] args) {
        // Test 1
        String text = "the sky is blue";
        runTestCase(text);

        // Test 2
        text = "  ab    ba  ";
        runTestCase(text);

        // Test 3
        text = "   ";
        runTestCase(text);

        // Test 4
        text = "";
        runTestCase(text);

        // Test 5
        text = "   the    sky    is    blue   ";
        runTestCase(text);
    }
}