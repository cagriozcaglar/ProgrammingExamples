/**
 For a web developer, it is very important to know how to design a web page's size. So, given a specific rectangular web
 page’s area, your job by now is to design a rectangular web page, whose length L and width W satisfy the following
 requirements:
 1. The area of the rectangular web page you designed must equal to the given target area.
 2. The width W should not be larger than the length L, which means L >= W.
 3. The difference between length L and width W should be as small as possible.

 You need to output the length L and the width W of the web page you designed in sequence.

 Example:
 Input: 4
 Output: [2, 2]
 Explanation: The target area is 4, and all the possible ways to construct it are [1,4], [2,2], [4,1].
 But according to requirement 2, [1,4] is illegal; according to requirement 3,  [4,1] is not optimal compared to [2,2]. So the length L is 2, and the width W is 2.

 Note:
 1. The given area won't exceed 10,000,000 and is a positive integer.
 2. The web page's width and length you designed must be positive integers.
*/

import java.util.*;  // Used for Arrays.toString() for printing Array contents

public class ConstructTheRectangle{

    /**
     * Solution 1: Correct, optimal, and very few lines
     * Given the area of a rectangle, contruct the length and width based on the following constraints:
     * 1. The area of the rectangular web page you designed must equal to the given target area.
     * 2. The width W should not be larger than the length L, which means L >= W.
     * 3. The difference between length L and width W should be as small as possible.
     * @param area
     * @return
     */
    public static int[] constructRectangle1(int area) {
        // Start with most-balanced (width, length) combination
        int width = (int)Math.sqrt(area);
        // Decrease width until it is a factor of area. Once a factor, you found width value, exit the loop
        while(area % width != 0){
            width--;
        }
        // Return (length, width) pair in one line, with no unnecessary array index assigment
        // Syntax hint: You can declare and instantiate an array in the same line as below.
        return new int[] {(area / width), width };
    }

    /**
     * Solution 2: Correct, optimal, but too many lines
     * Given the area of a rectangle, contruct the length and width based on the following constraints:
     * 1. The area of the rectangular web page you designed must equal to the given target area.
     * 2. The width W should not be larger than the length L, which means L >= W.
     * 3. The difference between length L and width W should be as small as possible.
     * @param area
     * @return int[] array: containing length, width
     */
    public static int[] constructRectangle2(int area) {
        // Output variable returning an array of two integers: length and width
        int[] dimensions = new int[2];
        // Square root of area is the number that makes width and length closest to each other
        int sqrtValue = (int)(Math.sqrt((double)area));
        // Decrease width until it is a factor of area. Once a factor, you found width value, exit the loop
        for(int width = sqrtValue; width >= 1; width--){
            if(area % width == 0){
                dimensions[0] = area / width;
                dimensions[1] = width;
                break;
            }
        }
        return dimensions;
    }

    /**
     * Run a test case with all solutions
     * @param area
     */
    public static void runTestCase(int area){
        System.out.println( "Answer with solution 1: " + Arrays.toString(constructRectangle1(area)) );
        System.out.println( "Answer with solution 2: " + Arrays.toString(constructRectangle2(area)) );
    }

    public static void main(String[] args){
        // Test 1: Square number case: 4 = 2 * 2
        // Syntax hint: Use Arrays.toString(arrayVar) to print array contents, after importing java.util.* .
        // If you print the array using its name, it will return memory address, e.g. "[I@4e25154f" .
        int area = 4;
        runTestCase(area);

        // Test 2: 12 = 4 * 3
        area = 12;
        runTestCase(area);

        // Test 3: 24 = 6 * 4
        area = 24;
        runTestCase(area);

        // Test 4: Square root case: 100 = 10 * 10
        area = 100;
        runTestCase(area);

        // Test 5: Prime number casecase: 11 = 11 * 1
        area = 11;
        runTestCase(area);
    }
}