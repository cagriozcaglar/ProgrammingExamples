/**
 Given a string, find the first non-repeating character in it and return its index. If it doesn't exist, return -1.

 Examples:
 1. s = "leetcode"
    return 0.
 2. s = "loveleetcode",
    return 2.

 Note: You may assume the string contain only lowercase letters.
 */

class FirstUniqueCharacterInString {

    /**
     * Solution with two passes, with extra variable declaration, not space-efficient.
     * In the first pass, get the character counts. In the second pass, find the first character with count == 1.
     * Note: Given that the string contains only lowercase letters, there are 'z'-'a'+1 = 26 possible characters in total.
     *
     * @param s
     * @return
     */
    public static int firstUniqChar(String s) {
        // Character count array for letters 'a' to 'z'
        int[] charCount = new int['z' - 'a' + 1];
        int charIndex;

        // First pass: Get character counts
        for (int i = 0; i < s.length(); i++) {
            charIndex = s.charAt(i) - 'a';
            charCount[charIndex] = charCount[charIndex] + 1;
        }

        // Second pass: Find the first unique character
        for (int i = 0; i < s.length(); i++) {
            charIndex = s.charAt(i) - 'a';
            if (charCount[charIndex] == 1) {
                return i;
            }
        }

        // If there is no unique character, return -1
        return -1;
    }

    /**
     * Solution with two passes, with no extra variable declaration, space-efficient.
     * In the first pass, get the character counts. In the second pass, find the first character with count == 1.
     * Note: Given that the string contains only lowercase letters, there are 'z'-'a'+1 = 26 possible characters in total.
     * Note: For increasing a value in array, "array[index]++" is sufficient, "array[index] = array[index] + 1" is not
     * needed.
     *
     * @param s
     * @return
     */
    public static int firstUniqCharOptimized(String s) {
        // Character counts for 'a'..'z'
        int[] frequency = new int[26];

        // First pass: Get character counts
        for (int i = 0; i < s.length(); i++)
            frequency[s.charAt(i) - 'a']++;

        // Second pass: Find the first unique character
        for (int i = 0; i < s.length(); i++)
            if (frequency[s.charAt(i) - 'a'] == 1)
                return i;

        // If there is no unique character, return -1
        return -1;
    }

    /**
     * Run a test case with all solutions
     * @param s
     */
    public static void runTestCase(String s) {
        System.out.println(s + ": " + firstUniqChar(s));
        System.out.println(s + ": " + firstUniqCharOptimized(s));
    }

    public static void main(String[] args) {
        // Test 1
        String word = "leetcode";
        runTestCase(word);

        // Test 2: Case when there is no unique / non-repeating character in the string
        word = "abcabc";
        runTestCase(word);

        // Test 3: Empty string: No character appears only once
        word = "";
        runTestCase(word);
    }
}