/**
 Given two lists Aand B, and B is an anagram of A. B is an anagram of A means B is made by randomizing the order of the elements in A.
 We want to find an index mapping P, from A to B. A mapping P[i] = j means the ith element in A appears in B at index j.
 These lists A and B may contain duplicates. If there are multiple answers, output any of them.

 Example:
  A = [12, 28, 46, 32, 50]
  B = [50, 12, 32, 46, 28]
  We should return
  [1, 4, 3, 2, 0]
  as P[0] = 1 because the 0th element of A appears at B[1], and P[1] = 4 because the 1st element of A appears at B[4], and so on.

 Note:
  1) A, B have equal lengths in range [1, 100].
  2) A[i], B[i] are integers in range [0, 10^5].
*/

import java.util.*;

public class FindAnagramMappings {
    /**
     * Given two integer arrays A and B, composed of same values in different order, return an index mapping P from array
     * A to B, such that P[i] = j if A[i] appears in B[j].
     *
     * Keep a map from values of B to the vector of indices, using HashMap<Integer, Vector<Integer>>. Iterate over A,
     * and map each A value to its index in B.
     *
     * @param A
     * @param B
     * @return
     */
    public static int[] findAnagramMappings(int[] A, int[] B) {
        int[] indexMap = new int[A.length];

        // Iterate over B, and map values of B to vector of its indices
        HashMap<Integer, Vector<Integer> > valueToIndexMap = new HashMap<Integer, Vector<Integer> >();
        for(int i = 0; i < B.length; i++) {
            if(valueToIndexMap.containsKey(B[i])){
                valueToIndexMap.get(B[i]).add(i);
            } else {
                Vector<Integer> newVector = new Vector<Integer>();
                newVector.add(i);
                valueToIndexMap.put(B[i], newVector);
            }
        }

        // Iterate over A, and indices of A value to the index of the same value in B
        for(int j = 0; j < A.length; j++) {
            indexMap[j] = valueToIndexMap.get(A[j]).remove(0);
        }

        return indexMap;
    }

    /**
     * Run test case given two arrays
     * @param array1
     * @param array2
     */
    public static void runTestCase(int[] array1, int[] array2) {
        int[] anagramMapping = findAnagramMappings(array1, array2);
        System.out.println("Array 1: " + Arrays.toString(array1));
        System.out.println("Array 2: " + Arrays.toString(array2));
        System.out.println("Anagram mapping: " + Arrays.toString(anagramMapping));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        int[] array1 = new int[] {12, 28, 46, 32, 50};
        int[] array2 = new int[] {50, 12, 32, 46, 28};
        runTestCase(array1, array2);

        // Test 2
        int[] array3 = new int[] {10, 10, 10, 10, 10};
        int[] array4 = new int[] {10, 10, 10, 10, 10};
        runTestCase(array3, array4);
    }
}