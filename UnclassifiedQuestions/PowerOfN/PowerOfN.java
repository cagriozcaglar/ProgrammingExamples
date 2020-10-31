/**
 Given an integer, write a function to determine if it is a power of 2.
 Given an integer, write another function to determine if it is a power of 4.
 */

class PowerOfN {

    /******
     * Power of 2
     ******/

    /**
     * Power of 2: Bit manipulation. Fast.
     * Short version with bit manipulation.
     * @param n
     * @return
     */
    public static boolean isPowerOfTwo(int n) {
        // Note: (n & (n-1) == 0) is the test for power of 2, but it is not enough.
        // Check the edge cases:
        // 1) if n <= 0 (0 and negatives), n is not a power of 2.
        // 2) if n == 1, n is a power of 2. (2^0)
        // 3) Otherwise, run (n & (n-1) == 0) test.
        return (n>0) ? ( (n & (n-1)) == 0 ) : false;
    }

    /**
     * Power of 2: Repeated division
     * Long version with repeated division
     * @param n
     * @return
     */
    public static boolean isPowerOfTwoLonger(int n){
        // Check edge cases first:
        // 1) if n <= 0 (0 and negatives), n is not a power of 2.
        if(n <= 0){
            return false;
        }
        // 2) if n == 1, n is a power of 2. (2^0)
        if(n == 1){
            return true;
        }
        // 3) Otherwise, keep dividing by 2, check if divisible, until you reach 1.
        while(n%2 == 0){
            n = n / 2;
            if(n == 1){
                return true;
            }
        }
        // If while loop is exited before returning true, n is not a power of 2, and return false.
        return false;
    }

    /******
     * Power of 4
     ******/

    /**
     * Power of 4 : Long version
     * @param num
     * @return
     */
    // Logic: Examples of powers of 4 are:
    // 4:      100
    // 16:   10000
    // 64: 1000000
    // Pattern: Starting from the right-most bit, powers of 4 start with "even number of 0s", followed by exactly one 1,
    // followed by all 0s.
    // To make this check: we check the last bit of the number. If it is 0, we continue and increase the number of 0s before 1.
    // If it is 1, we check the following two conditions:
    // 1) The number of 0s before 1 has to be even.
    // 2) The rest of the number has to be 0.
    public static boolean isPowerOfFour(int num) {
        // Number of zeros before one: We will check if this is an even number
        int numberOfZerosBeforeOne = 0;
        // Boolean value indicating whether a 1 has been seen so far
        boolean oneIsSeen = false;
        // While number is not zero, continue
        while(num != 0){
            // If the bit is 0, increase numberOfZerosBeforeOne by 1, shift the number by 1 bit, continue to next iteration
            if( (num & 1) == 0){
                numberOfZerosBeforeOne++;
                num = num >> 1;
                continue;
            } // If the bit is 1, set oneIsSeen to true, shift the number by 1 bit, exit out of while loop for final checks
            else {
                oneIsSeen = true;
                num = num >> 1;
                break;
            }
        }
        // At the end, check the following:
        // 1) The number of 0s before 1 has to be even.
        // 2) The rest of the number has to be 0.
        // 3) Also check if the number is 0. (Why? Because, if the right-most bit of the original number is 1,
        // first two conditions above are satisfied, but the original number if not a power of 4.
        return ((numberOfZerosBeforeOne %2 == 0) && oneIsSeen && (num == 0)) ? true : false;
    }

    /**
     * Power of 4: Short version 1
     * First two conditions check whether the number is power of 2: (num > 0) && ( (num & (num-1)) == 0 ).
     * Last condition removes the numbers that are power of 2, but not power of 4.
     * @param num
     * @return
     * 0101
     */
    public static boolean isPowerOfFourShort1(int num) {
        // First two conditions check whether the number is a power of 2
        // Third condition: 0x55555555 is to get rid of those power of 2 but not power of 4
        // so that the single 1 bit always appears at the odd position.
        return (num > 0) && ( (num & (num-1)) == 0 ) && ( (num & 0x55555555) != 0);
    }

    /**
     * Power of 4: Short version 2
     * First two conditions check whether the number is power of 2: (num > 0) && ( (num & (num-1)) == 0 ).
     * Last condition: The only "1" should always located at the odd position, e.g, 4^0 = 1, 4^1 = 100, 4^2=10000.
     * So we can use "(num & 0x55555555) == num" to check if "1" is located at the odd position.
     * @param num
     * @return
     */
    public static boolean isPowerOfFourShort2(int num) {
        // The only "1" should always located at the odd position, e.g, 4^0 = 1, 4^1 = 100, 4^2=10000.
        // So we can use "num & 0x55555555==num" to check if "1" is located at the odd position.
        return (num > 0) && ((num & (num - 1)) == 0) && ((num & 0x55555555) == num);
    }

    /**
     * Power of 4: Short version 3
     * First two conditions check whether the number is power of 2: (num > 0) && ( (num & (num-1)) == 0 ).
     * Thirs condition "(num - 1) % 3 == 0" ensures the number is also a power of 4. Check proof in the comments below.
     * @param num
     * @return
     */
    // First two conditions "(num > 0) && (num & (num - 1)) == 0" check for powers of 2.
    // Third condition checks for power of 4. Why?
    // Given that the number is of the form 2^n, it can either be of the form 2^(n=2k) or 2^(n=2k+1).
    // If n=2k => 2^{2k} - 1 = 4^k -1 = (4-1) * (4^(k-1) + 4^(k-2) + 4^(k-3) + ..... + 4^1 + 4^0) == 0 (mod 3)
    // If n=2k+1 => 2^{2k+1} - 1 = (2 * 4^k) - 1
    public static boolean isPowerOfFourShort3(int num) {
        return (num > 0) && (num & (num - 1)) == 0 && (num - 1) % 3 == 0;
    }

    public static void main(String[] args){

        System.out.println("Testing power of 2");

        // Test 1.1: Power of 2 - True
        int x = 8;
        System.out.println(x + ": " + isPowerOfTwo(x));
        System.out.println(x + ": " + isPowerOfTwoLonger(x));

        // Test 1.2: Power of 2 - False
        x = 9;
        System.out.println(x + ": " + isPowerOfTwo(x));
        System.out.println(x + ": " + isPowerOfTwoLonger(x));

        // Test 1.3: Power of 2 - Edge case (1) - True
        x = 1;
        System.out.println(x + ": " + isPowerOfTwo(x));
        System.out.println(x + ": " + isPowerOfTwoLonger(x));

        // Test 1.4: Power of 2 - Edge case (0) - False
        x = 0;
        System.out.println(x + ": " + isPowerOfTwo(x));
        System.out.println(x + ": " + isPowerOfTwoLonger(x));

        // Test 1.5: Power of 2 - Edge case (negative) - False
        x = -12;
        System.out.println(x + ": " + isPowerOfTwo(x));
        System.out.println(x + ": " + isPowerOfTwoLonger(x));

        System.out.println("Testing power of 4");

        // Test 2.1: Power of 4 - True
        x = 16;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));

        // Test 2.2: Power of 4 - False
        x = 22;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));

        // Test 2.3: Power of 4 - Edge case (power of 2, but not 4) - False
        x = 32;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));

        // Test 2.4: Power of 4 - Edge case (1) - True
        x = 1;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));

        // Test 2.5: Power of 4 - Edge case (0) - False
        x = 0;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));

        // Test 2.6: Power of 4 - Edge case (negative) - False
        x = -15;
        System.out.println(x + ": " + isPowerOfFour(x));
        System.out.println(x + ": " + isPowerOfFourShort1(x));
        System.out.println(x + ": " + isPowerOfFourShort2(x));
        System.out.println(x + ": " + isPowerOfFourShort3(x));
    }
}