/**
 You are given a string representing an attendance record for a student. The record only contains the following three
 characters:

 'A' : Absent.
 'L' : Late.
 'P' : Present.

 A student could be rewarded if his attendance record doesn't contain more than one 'A' (absent) or more than two
 continuous 'L' (late). You need to return whether the student could be rewarded according to his attendance record.

 Example 1:
 Input: "PPALLP"
 Output: True

 Example 2:
 Input: "PPALLL"
 Output: False
 */

/**
 * NOTE: Practice Java Regular expressions here: https://docs.oracle.com/javase/7/docs/api/java/util/regex/Pattern.html
 * Java Regular Expression Examples: http://www.vogella.com/tutorials/JavaRegularExpressions/article.html
 */

public class StudentAttendanceRecord1 {
    /**
     * Solution: Use regular expression
     * 1) String does not contain more than one 'A': !(s.matches(".*A.*A.*"))
     * 2) String does not contain more than 2 continuous 'L': This is equivalent to checking if the string does not have
     * 3 consecutive L's: !(s.matches(".*(LLL)+.*"))
     * @param s
     * @return
     */
    public static boolean checkRecord1(String s) {
        return !(s.matches(".*A.*A.*")) && !(s.matches(".*LLL.*"));
    }

    /**
     * Solution: Use regular expression, same as above, but both conditions OR'ed into one condition
     * 1) String does not contain more than one 'A': !(s.matches(".*A.*A.*"))
     * 2) String does not contain more than 2 continuous 'L': This is equivalent to checking if the string does not have
     * 3 consecutive L's: !(s.matches(".*(LLL)+.*"))
     * Both conditions OR'ed: !s.matches(".*A.*A.*|.*(LLL)+.*");
     * @param s
     * @return
     */
    public static boolean checkRecord2(String s) {
        return !s.matches(".*A.*A.*|.*(LLL)+.*");
    }

    /**
     * Solution: Without regular expression, use string search methods ("contains", "indexOf", "lastIndexOf") to
     * check the conditions.
     * 1) String does not contain more than one 'A': The string can have 0 or 1 'A'. Only in these two cases,
     * s.indexOf("A") and s.lastIndexOf("A") returns the same value:
     *   a) If there is no 'A': s.indexOf("A") and s.lastIndexOf("A") returns -1.
     *   b) If there is 1 'A': s.indexOf("A") and s.lastIndexOf("A") returns -1.
     * Therefore, only in these two cases, and in no other case, (s.indexOf("A") == s.lastIndexOf("A")) holds.
     * 2) String does not contain more than 2 continuous 'L': This is equivalent to the condition of the string not
     * containing "LLL": This can be written using contains method: !(s.contains("LLL"))
     *
     * NOTE: String methods used in this solution:
     * 1) int indexOf(char ch): This method returns the index within this string of the first occurrence of the
     * specified character or -1, if the character does not occur. More: https://www.tutorialspoint.com/java/java_string_indexof.htm
     * 2) int lastIndexOf(int ch): This method returns the index of the last occurrence of the character in the character
     * sequence represented by this object that is less than or equal to fromIndex, or -1 if the character does not occur
     * before that point. More: https://www.tutorialspoint.com/java/java_string_lastindexof.htm
     * 3) boolean contains(CharSequence s): The java.lang.String.contains() method returns true if and only if this
     * string contains the specified sequence of char values. More: https://www.tutorialspoint.com/java/lang/string_contains.htm
     *
     * @param s
     * @return
     */
    public static boolean checkRecord3(String s) {
        return (s.indexOf("A") == s.lastIndexOf("A")) && !(s.contains("LLL"));
    }

    /**
     * Run a test case with all solutions
     * @param s
     */
    public static void runTestCase(String s){
        System.out.println("checkRecord1(\"" + s + "\"): " + checkRecord1(s));
        System.out.println("checkRecord2(\"" + s + "\"): " + checkRecord2(s));
        System.out.println("checkRecord3(\"" + s + "\"): " + checkRecord3(s));
    }

    public static void main(String[] args){
        // Test 1: "PPALLP" (true)
        String s = "PPALLP";
        runTestCase(s);

        // Test 2: "PPALLL" (false, breaks the rule about at most 2 consecutive Ls)
        s = "PPALLL";
        runTestCase(s);

        // Test 3: "APALA" (false, breaks the rule about not having more than 1 A)
        s = "APALA";
        runTestCase(s);
    }
}