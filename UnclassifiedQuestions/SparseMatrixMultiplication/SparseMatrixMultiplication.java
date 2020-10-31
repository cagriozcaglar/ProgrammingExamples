/**
 Given two sparse matrices A and B, return the result of AB.
 You may assume that A's column number is equal to B's row number.

 Example:
 A = [[ 1, 0, 0],
      [-1, 0, 3]]

 B = [[ 7, 0, 0 ],
      [ 0, 0, 0 ],
      [ 0, 0, 1 ]]

      |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
 AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                   | 0 0 1 |

 */

import java.util.*;

public class SparseMatrixMultiplication {

    /**
     * A \in R^{n x m}, B \in R^{m x p} => C \in R^{n x p}
     * c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
     * For c_{ij} calculation, i-th row of A and j-th column of B are multiplied (dot product).
     *
     * Observation: When A_{ik} is 0, there is no need to compute B_{kj}. So, we switch the inner two loops in the Naive
     * solution below, and add a condition to check if A_{ik} is 0. Similarly, when B_{kj} is zero, there is no need to
     * mutliply A_{ik} and B_{kj}
     *
     * Time complexity: In theory, since the matrix is sparse, time complexity is ~O(n^2). But in practice, this method
     * turns out to be slower, due to too many condition checks.
     *
     * @param A
     * @param B
     * @return
     */
    public static int[][] sparseMatrixMultiplicationEfficient1(int[][] A, int[][] B) {
        // Error checks
        if(A == null ||                 // A is null
           B == null ||                 // B is null
           A.length == 0 ||             // A is empty
           B.length == 0 ||             // B is empty
           A[0].length != B.length) {   // Number of columns in A is not equal to number of rows in B
            return new int[0][0]; // Empty 2D-array
        }

        int n = A.length;
        int m = B.length;
        int p = B[0].length;

        // Initialize output matrix
        int[][] C = new int[n][p];

        /**
         * Different part
         */
        // Create multiplication output matrix
        // Iterate over rows of A
        for(int i = 0; i < A.length; i++) {
            // Iterate over columns of B
            for(int k = 0; k < A[0].length; k++) {
                if(A[i][k] != 0) {
                    // Init sum c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj} to 0 first
                    // Iterate over columns of A: multiply row A[i,] with column B[,j]
                    for (int j = 0; j < B[0].length; j++) {
                        if(B[k][j] != 0) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
        return C;
    }

    /**
     * A \in R^{n x m}, B \in R^{m x p} => C \in R^{n x p}
     * c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
     * For c_{ij} calculation, i-th row of A and j-th column of B are multiplied (dot product).
     *
     * Idea: For array A, create a map from row index of A to another map which maps index at column index of A the value at
     * the (row,column) pair. That is, for A[i][j] != 0, we create Map(i, Map(j, A[i][j]) ). Same for B as well.
     *
     * Similar to this one: http://buttercola.blogspot.com/2016/01/leetcode-sparse-matrix-multiplication.html
     * 4th idea: https://discuss.leetcode.com/topic/99614/one-way-to-optimize-multiplication-of-two-sparse-matrices
     *
     * Time complexity: In theory, since the matrix is sparse, time complexity is ~O(n^2). But in practice, this method
     * turns out to be slower, due to HashMap creation steps and key existence checks.
     *
     * @param A
     * @param B
     * @return
     */
    public static int[][] sparseMatrixMultiplicationEfficient2(int[][] A, int[][] B) {
        // Error checks
        if(A == null ||                      // A is null
                B == null ||                 // B is null
                A.length == 0 ||             // A is empty
                B.length == 0 ||             // B is empty
                A[0].length != B.length) {   // Number of columns in A is not equal to number of rows in B
            return new int[0][0]; // Empty 2D-array
        }

        int n = A.length;
        int m = B.length;
        int p = B[0].length;

        // Initialize output matrix
        int[][] C = new int[n][p];

        /**
         * Different part
         */

        // Step 1: Convert the sparse A to dense format
        HashMap<Integer, HashMap<Integer, Integer>> denseA = new HashMap<>();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (A[i][j] != 0) {
                    if (!denseA.containsKey(i)) {
                        denseA.put(i, new HashMap<>());
                    }
                    denseA.get(i).put(j, A[i][j]); // HashMap collision in the inner map is not possible
                }
            }
        }

        // Step 2: Convert the sparse B to dense format
        HashMap<Integer, HashMap<Integer, Integer>> denseB = new HashMap<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                if (B[i][j] != 0) {
                    if (!denseB.containsKey(i)) {
                        denseB.put(i, new HashMap<>());
                    }
                    denseB.get(i).put(j, B[i][j]);
                }
            }
        }

        // Step 3: Calculate the denseA * denseB
        for(int i : denseA.keySet()) {
            for(int j : denseA.get(i).keySet()) {
                if(!denseB.containsKey(j)) {
                    continue;
                }
                for(int k : denseB.get(j).keySet()) {
                    C[i][k] += denseA.get(i).get(j) * denseB.get(j).get(k);
                }
            }
        }

        return C;
    }

    /**
     * A \in R^{n x m}, B \in R^{m x p} => C \in R^{n x p}
     * c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
     * For c_{ij} calculation, i-th row of A and j-th column of B are multiplied (dot product).
     *
     * Idea: Find nonzero *rows* of A, and nonzero *columns* of B (Because at each C[i][j] calculation, i-th row of A and
     * j-th column of B is multiplied: c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}). Using these two sets, optimize final
     * multiplication in the for loops.
     *
     * Similar to this one: https://discuss.leetcode.com/topic/99614/one-way-to-optimize-multiplication-of-two-sparse-matrices
     *
     * Time complexity: In theory, since the matrix is sparse, time complexity is ~O(n^2). But in practice, this method
     * turns out to be slower, due to HashSet creation steps and key existence checks.
     *
     * @param A
     * @param B
     * @return
     */
    public static int[][] sparseMatrixMultiplicationEfficient3(int[][] A, int[][] B) {
        // Error checks
        if(A == null ||                      // A is null
                B == null ||                 // B is null
                A.length == 0 ||             // A is empty
                B.length == 0 ||             // B is empty
                A[0].length != B.length) {   // Number of columns in A is not equal to number of rows in B
            return new int[0][0]; // Empty 2D-array
        }

        int n = A.length;
        int m = B.length;
        int p = B[0].length;

        // Initialize output matrix
        int[][] C = new int[n][p];

        /**
         * Different part
         */
        HashSet<Integer> nonzeroRowsA = new HashSet<Integer>();
        HashSet<Integer> nonzeroColumnsB = new HashSet<Integer>();

        // Find nonzero *rows* of A
        for(int i = 0; i < A.length; i++) {
            for(int j = 0; j < A[0].length; j++) {
                if(A[i][j] != 0) {// Mark nonzero row of A
                    nonzeroRowsA.add(i);
                    break;  // Exits the for-loop for the *column* iteration, because we are marking nonzero *rows* of A
                }
            }
        }

        // Find nonzero *columns* of B.
        // Note: Iterates over the *columns* in the *outer* for loop, because we want to early exit inner *row* iteration
        for(int j = 0; j < B[0].length; j++) { // Iterate over columns of B
            for(int i = 0; i < B.length; i++) {
                if(B[i][j] != 0) { // Mark nonzero column of B
                    nonzeroColumnsB.add(j);
                    break;
                }
            }
        }

        // Multiplication with checks on nonzero rows of A and nonzero columns of B
        for(int i = 0; i < A.length; i++) { // All rows of A
            if(nonzeroRowsA.contains(i)) {  // Check if row of A is nonzero
                for(int j = 0; j < B[0].length; j++) { // All columns of B
                    if(nonzeroColumnsB.contains(j)) { // Check if column of B is nonzero
                        // Calculation of C[i][j]: c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
                        for(int k = 0; k < A[0].length; k++) {
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }

        return C;
    }

    /**
     * A \in R^{n x m}, B \in R^{m x p} => C \in R^{n x p}
     * c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
     * For c_{ij} calculation, i-th row of A and j-th column of B are multiplied (dot product).
     *
     * Time complexity: O(n * m * p), where A \in R^{n x m}, B \in R^{m x p}  => O(n^3) => inefficient
     *
     * @param A
     * @param B
     * @return
     */
    public static int[][] sparseMatrixMultiplicationNaiveMethod(int[][] A, int[][] B) {
        // Error checks
        if(A == null ||                 // A is null
           B == null ||                 // B is null
           A.length == 0 ||             // A is empty
           B.length == 0 ||             // B is empty
           A[0].length != B.length) {   // Number of columns in A is not equal to number of rows in B
            return new int[0][0]; // Empty 2D-array
        }

        int n = A.length;
        int m = B.length;
        int p = B[0].length;

        // Initialize output matrix
        int[][] C = new int[n][p];

        // Create multiplication output matrix
        // Iterate over rows of A
        for(int i = 0; i < A.length; i++) {
            // Iterate over columns of B
            for(int j = 0; j < B[0].length; j++) { // CAREFUL: B[0].length, instead of B.length. Because we need columns of B.
                C[i][j] = 0; // Init C[i][j] to 0 first
                // Iterate over columns of A: multiply row A[i,] with column B[,j]
                // sum c_{ij} = sum_{k=1}^{m} a_{ik} * b_{kj}
                for(int k = 0; k < A[0].length; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return C;
    }

    public static void runTestCase(int[][] A, int[][] B) {
        long startTime;
        long endTime;

        // NOTE: Use Arrays.deepToString() for printing 2D-arrays, instead of Arrays.toString().

        // Efficient Method
        startTime = System.nanoTime();
        System.out.println("Array " + Arrays.deepToString(A) + " times Array " + Arrays.deepToString(B) + " is equal to " +
                            Arrays.deepToString(sparseMatrixMultiplicationEfficient1(A, B)) );
        endTime = System.nanoTime();
        System.out.println("Time for Efficient Method: " + (endTime - startTime));

        // Efficient Method 2
        startTime = System.nanoTime();
        System.out.println("Array " + Arrays.deepToString(A) + " times Array " + Arrays.deepToString(B) + " is equal to " +
                            Arrays.deepToString(sparseMatrixMultiplicationEfficient2(A, B)) );
        endTime = System.nanoTime();
        System.out.println("Time for Efficient Method 2: " + (endTime - startTime));

        // Efficient Method 3
        startTime = System.nanoTime();
        System.out.println("Array " + Arrays.deepToString(A) + " times Array " + Arrays.deepToString(B) + " is equal to " +
                            Arrays.deepToString(sparseMatrixMultiplicationEfficient3(A, B)) );
        endTime = System.nanoTime();
        System.out.println("Time for Efficient Method 3: " + (endTime - startTime));

        // Naive Method
        startTime = System.nanoTime();
        System.out.println("Array " + Arrays.deepToString(A) + " times Array " + Arrays.deepToString(B) + " is equal to " +
                            Arrays.deepToString(sparseMatrixMultiplicationNaiveMethod(A, B)) );
        endTime = System.nanoTime();
        System.out.println("Time for Naive Method: " + (endTime - startTime));

        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        int[][] A = new int[][] { { 1, 0, 0},
                                  {-1, 0, 3} };
        int[][] B = new int[][] { { 7, 0, 0 },
                                  { 0, 0, 0 },
                                  { 0, 0, 1 } };
        runTestCase(A, B);
    }
}