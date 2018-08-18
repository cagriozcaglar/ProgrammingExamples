/**
 The Hamming distance between two integers is the number of positions at which the corresponding bits are different.
 Given two integers x and y, calculate the Hamming distance.

 Note:
 0 ≤ x, y < 231.

 Example:
 Input: x = 1, y = 4
 Output: 2
 Explanation:
 1   (0 0 0 1)
 4   (0 1 0 0)
 ↑   ↑
 The above arrows point to positions where the corresponding bits are different.
 */

import java.util.*;

class HammingDistance {
    /**
     * Method 1: XOR the integers, count the number of bits that are 1.
     * @param x
     * @param y
     * @return
     */
    public static int hammingDistance1(int x, int y) {
        int count = 0;
        int differenceBits = x ^ y;
        while(differenceBits != 0){
            // Mask all bits except first bit, check if it is non-zero (meaning 1)
            // Notes:
            // 1) Be careful, bitwise and is ** & **, not ** && ** (this is conditional AND). (Common mistake).
            // 2) Parantheses around "(differenceBits & 1)" is needed. If you remove parantheses and use "differenceBits & 1 != 0",
            // it returns the following error: "error: bad operand types for binary operator '&'".
            if( (differenceBits & 1) != 0){
                count++;
            }
            differenceBits = differenceBits >> 1; // Shift the number by 1 to right, to check the next bit
        }
        return count;
    }

    /**
     * Method 2: XOR the integer, use Integer.bitCount() method to count the number of 1s.
     * @param x
     * @param y
     * @return
     */
    public static int hammingDistance2(int x, int y) {
        return Integer.bitCount(x^y);
    }

    public static void main(String[] args){
        // Timers for profiling
        long startTime;
        long endTime;

        // Test 1
        int x = 1;
        int y = 4;

        // Hamming distance 1
        startTime = System.nanoTime();
        System.out.println( hammingDistance1(x,y) );
        endTime = System.nanoTime();
        System.out.println("Time for hammingDistance1: " + (endTime - startTime));
        // Time for hammingDistance1: 159486

        // Hamming distance 2: faster
        startTime = System.nanoTime();
        System.out.println( hammingDistance2(x,y) );
        endTime = System.nanoTime();
        System.out.println("Time for hammingDistance2: " + (endTime - startTime));
        // Time for hammingDistance2: 38693

        // Test 2
        x = 67;
        y = 92;

        // Hamming distance 1
        startTime = System.nanoTime();
        System.out.println( hammingDistance1(x,y) );
        endTime = System.nanoTime();
        System.out.println("Time for hammingDistance1: " + (endTime - startTime));
        // Time for hammingDistance1: 15940

        // Hamming distance 2: faster
        startTime = System.nanoTime();
        System.out.println( hammingDistance2(x,y) );
        endTime = System.nanoTime();
        System.out.println("Time for hammingDistance2: " + (endTime - startTime));
        // Time for hammingDistance2: 15255

    }
};