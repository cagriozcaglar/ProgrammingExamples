/**
 * Consider an array of numeric strings, "unsorted", where each string is a positive number with anywhere from 1 to 10^6 digits.
 * Sort the array's elements in non-decreasing (i.e., ascending) order of their real-world integer values and print each
 * element of the sorted array on a new line.
 * Note: Each string is guaranteed to represent a positive integer without leading zeros.
 * <p>
 * ****
 * Sample Input:
 * 31415926535897932384626433832795
 * 1
 * 3
 * 10
 * 3
 * 5
 * <p>
 * Sample Output:
 * 1
 * 3
 * 3
 * 5
 * 10
 * 31415926535897932384626433832795
 * <p>
 * Explanation:
 * The initial array of strings is unsorted=[31415926535897932384626433832795, 1, 3, 10, 3, 5]. When we order each string
 * by the real-world integer value it represents, we get: 1 <= 3 <= 3 <= 5 <= 10 <= 31415926535897932384626433832795.
 * ****
 */


import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class BigIntegerSorting {

    // Inner class for implementaing Big Integer String comparator
    // Note the following:
    // 1) This class has to be static (because non-static variable this cannot be referenced from a static context)
    static class BigIntegerStringComparatorInnerClass implements Comparator<String> {
        public int compare(String s1, String s2) {
            // Sort in increasing order
            // Logic of big integer string sorting (same in all three solutions)
            // 1. Check string lengths: if one string is longer than the other, that string is the larger integer.
            // 2. If the lengths of the strings are the same, compare the strings in lexicographical (alphabetical) order.
            // Based on the output of s1.compareTo(s2), the order of two strings are returned as follows:
            // 2.1. If s1.compareTo(s2) < 0 => integer value of s1 is smaller than integer value of s2.
            // 2.2. If s1.compareTo(s2) > 0 => integer value of s1 is larger than integer value of s2.
            // 2.3. If s1.compareTo(s2) == 0 => integer value of s1 is equal to integer value of s2.
            return (s1.length() < s2.length()) ? -1 : ((s1.length() > s2.length()) ? 1 : (s1.compareTo(s2)));
        }
    }

    public static void main(String[] args) {
        String[] unsorted = {"123", "321", "111", "1", "22", "432423", "5"};

        /**
         * Option 1: Sorting with an inner class Comparator
         */
        System.out.println("Sorting with an inner class Comparator");
        Arrays.sort(unsorted, new BigIntegerStringComparatorInnerClass());
        for (int i = 0; i < unsorted.length; i++) {
            System.out.println(unsorted[i]);
        }

        /**
         * Option 2: Sorting with an outer class Comparator
         */
        System.out.println("Sorting with an outer class Comparator");
        Arrays.sort(unsorted, new BigIntegerStringComparatorOuterClass());
        for (int i = 0; i < unsorted.length; i++) {
            System.out.println(unsorted[i]);
        }

        /**
         * Option 3: Sorting with an anonymous class (inline)
         */
        System.out.println("Sorting with an anonymous class (inline)");
        // Anonymous class for Big Integer sorting in the following line
        // Note that this Arrays.sort() operations sorts in-place, hence changes "unsorted" array
        Arrays.sort(unsorted, new Comparator<String>() {
                    public int compare(String s1, String s2) {
                        // Sort in increasing order
                        return (s1.length() < s2.length()) ? -1 : ((s1.length() > s2.length()) ? 1 : (s1.compareTo(s2)));
                    }
                }
        );
        for (int i = 0; i < unsorted.length; i++) {
            System.out.println(unsorted[i]);
        }
    }
}

// Outer class for implementing Big Integer String comparator.
// Note the following:
// 1) This class does not need to be static.
// 2) This class can not be public. (at most one public class in a java file)
class BigIntegerStringComparatorOuterClass implements Comparator<String> {
    public int compare(String s1, String s2) {
        // Sort in increasing order
        return (s1.length() < s2.length()) ? -1 : ((s1.length() > s2.length()) ? 1 : (s1.compareTo(s2)));
    }
}
