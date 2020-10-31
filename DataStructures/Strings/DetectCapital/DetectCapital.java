/**
 Given a word, you need to return whether the usage of capitals in it is right or not.
 We define the usage of capitals in a word to be right when one of the following cases holds:
 1. All letters in this word are capitals, like "USA".
 2. All letters in this word are not capitals, like "leetcode".
 3. Only the first letter in this word is capital if it has more than one letter, like "Google".
 Otherwise, we define that this word doesn't use capitals in a right way.

 Example 1:
 Input: "USA"
 Output: True

 Example 2:
 Input: "FlaG"
 Output: False

 Note: The input will be a non-empty word consisting of uppercase and lowercase latin letters.
 */

class DetectCapital {
    /**
     * Return whether a given word conforms to the rules of word capitalization:
     * 1. All letters in this word are capitals, like "USA".
     * 2. All letters in this word are not capitals, like "leetcode".
     * 3. Only the first letter in this word is capital if it has more than one letter, like "Google".
     * @param word
     * @return
     */
    public static boolean detectCapitalUse(String word) {
        // Handle empty string
        if(word.length() == 0){
            return true;
        }
        // 1. All letters in this word are capitals, like "USA".
        String allCapital = word.toUpperCase();
        // 2. All letters in this word are not capitals, like "leetcode".
        String allLowercase = word.toLowerCase();
        // 3. Only the first letter in this word is capital if it has more than one letter, like "Google".
        String onlyFirstCharacterCapital = word.substring(0,1).toUpperCase() + word.substring(1).toLowerCase();

        // Check if the string format conforms to one of the three cases
        return word.equals(allCapital) || word.equals(allLowercase) || word.equals(onlyFirstCharacterCapital);
    }

    /**
     * Detect capital, using regex, very short solution
     * 1. [A-Z]* => All capitals (0+ times A-Z)
     * 2. [a-z]* => All lowercase (0+ times a-z)
     * 3. [A-Z][a-z]* => Capital, followed by all lowercase (1 A-Z, followed by 0+ a-z)
     * @param word
     * @return
     */
    public static boolean detectCapitalUseRegex(String word) {
        return word.matches("[A-Z]*|[a-z]*|[A-Z][a-z]*");
    }

    /**
     * Run a test case with all solutions
     * @param word
     */
    public static void runTestCase(String word){
        System.out.println(word + ": " + detectCapitalUse(word));
        System.out.println(word + ": " + detectCapitalUseRegex(word));
    }

    public static void main(String[] args){
        // Test 1
        String word = "USA";
        runTestCase(word);

        // Test 2
        word = "leetcode";
        runTestCase(word);

        // Test 3
        word = "Alright";
        runTestCase(word);

        // Test 4
        word = "FlaG";
        runTestCase(word);

        // Test 5
        word = "";
        runTestCase(word);
    }
}