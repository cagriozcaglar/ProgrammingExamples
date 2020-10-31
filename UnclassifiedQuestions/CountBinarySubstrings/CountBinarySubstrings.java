/**
 Given a string s, count the number of non-empty (contiguous) substrings that have the same number of 0's and 1's, and
 all the 0's and all the 1's in these substrings are grouped consecutively.
 Substrings that occur multiple times are counted the number of times they occur.

 Example 1:
 Input: "00110011"
 Output: 6
 Explanation: There are 6 substrings that have equal number of consecutive 1's and 0's: "0011", "01", "1100", "10", "0011", and "01".
 Notice that some of these substrings repeat and are counted the number of times they occur.
 Also, "00110011" is not a valid substring because all the 0's (and 1's) are not grouped together.

 Example 2:
 Input: "10101"
 Output: 4
 Explanation: There are 4 substrings: "10", "01", "10", "01" that have equal number of consecutive 1's and 0's.

 Note:
 1) s.length will be between 1 and 50,000.
 2) s will only consist of "0" or "1" characters.
 */

import java .util.*; // For ArrayList

public class CountBinarySubstrings {

    /**
     * Solution 1: Use size of consecutive groups. The idea is summarized here: https://leetcode.com/problems/count-binary-substrings/solution/
     * 1) We can convert the string s into an array groups that represents the length of same-character contiguous blocks
     * within the string. For example, if s = "110001111000000", then groups = [2, 3, 4, 6].
     * 2) Let's try to count the number of valid binary strings between groups[i] and groups[i+1]. If we have groups[i] = 2,
     * groups[i+1] = 3, then it represents either "00111" or "11000". We clearly can make min(groups[i], groups[i+1])
     * valid binary strings within this string. Because the binary digits to the left or right of this string must change
     * at the boundary, our answer can never be larger.
     *
     * Algorithm:
     * 1) Create groups as defined above. The first element of s belongs in it's own group. From then on, each element
     * either doesn't match the previous element, so that it starts a new group of size 1; or it does match, so that the
     * size of the most recent group increases by 1.
     * 2) Take the sum of all min(groups[i-1], groups[i]).
     *
     * Time complexity: O(n)
     * Space complexity: O(n) , space used by groups
     *
     * @param s
     * @return
     */
    public static int countBinarySubstringsWithGroups(String s) {
        // Error checks
        if(s == null || s.length() == 0){
            return 0;
        }

        List<Integer> groups = new ArrayList<Integer>();
        int groupIndex = 0;
        groups.add(1); // at index 0

        int totalBinarySubstrings = 0;

        // Create group sizes
        for(int i = 1; i < s.length(); i++) {
            // 1. Different characters: add new group
            if(s.charAt(i-1) != s.charAt(i)) {
                groupIndex++;
                groups.add(groupIndex, 1);
            } else { // 2. Same characters: increment count in current group
                groups.set(groupIndex, groups.get(groupIndex)+1);
            }
        }

        // Get min(groups(i-1), groups(i)) sizes, and sum them to get the final count
        for(int i = 1; i <= groupIndex; i++) {
            totalBinarySubstrings += Math.min(groups.get(i-1), groups.get(i));
        }

        return totalBinarySubstrings;
    }

    /**
     * Solution 2: Same algorithm as in Solution 1, but we only remember prev = groups[-2] and curr = groups[-1].
     * The idea is summarized here: https://leetcode.com/problems/count-binary-substrings/solution/
     * We can amend Solution 1 to calculate the answer on the fly. Instead of storing groups, we will remember only
     * prev = groups[-2] and cur = groups[-1]. Then, the answer is the sum of min(prev, cur) over each different final
     * (prev, cur) we see.
     *
     * Time complexity: O(n)
     * Space complexity: O(1) (variables prev, curr, count)
     *
     * @param s
     * @return
     */
    public static int countBinarySubstringsLinearScan(String s) {
        // Error checks
        if(s == null || s.length() == 0){
            return 0;
        }

        // prev: groups[-2]
        int prev = 0;
        // curr: groups[-1]
        int curr = 1;
        // Total binary substrings
        int totalBinarySubstrings = 0;

        // Iterate over the array, compare elements s.charAt(i) and s.charAt(i-1) at each step
        for(int i = 1; i < s.length(); i++) {
            // 1. Same characters
            if(s.charAt(i) == s.charAt(i-1)) {
                curr++;
            } else { // 2. Different characters
                totalBinarySubstrings += Math.min(prev, curr);
                prev = curr; // Size of prev is now equal to size of curr
                curr = 1;    // Size of current is initially 1
            }
        }
        // CAREFUL: Add the size in the border of last pair of (prev, curr)
        totalBinarySubstrings += Math.min(prev, curr);

        return totalBinarySubstrings;
    }

    public static void runTestCase(String binaryString) {
        System.out.println("Number of binary substrings in \"" + binaryString + "\" with equal number of consecutive 0s and 1s is: " +
                            countBinarySubstringsWithGroups(binaryString));
        System.out.println("Number of binary substrings in \"" + binaryString + "\" with equal number of consecutive 0s and 1s is: " +
                            countBinarySubstringsLinearScan(binaryString));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        String binaryString = "00110011";
        runTestCase(binaryString);

        // Test 2
        binaryString = "10101";
        runTestCase(binaryString);
    }
}