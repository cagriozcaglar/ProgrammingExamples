/**
 Given a positive integer num, write a function which returns True if num is a perfect square else False.
 Note: Do not use any built-in library function such as sqrt.

 Example 1:
 Input: 16
 Returns: True

 Example 2:
 Input: 14
 Returns: False
 */
// TODO: Fix Solution 4, which uses exponential identity to check valid perfect square.

class ValidPerfectSquare {
    /**
     * Solution 1: Binary search
     * Running time: O(log(num))
     * Note 1: The limit variables "low", "high", "current" are of type long, instead of 0. This is because we want
     * to avoid "current * current" from overflow (this does happen, did happen).
     * Note 2: Be careful of setting low / high limits to above / below current, otherwise it results in infinite loop.
     * @param num
     * @return
     */
    public static boolean isPerfectSquare1(int num) {
        // The limit variables "low", "high", "current" are of type long, instead of 0. This is because we want
        // to avoid "current * current" from overflow (this does happen, did happen).
        long low = 1;
        long high = num;
        while(low <= high){
            long current = (low + high) / 2;
            long sqr = current * current;
            if(sqr < num){
                // Increment, because otherwise, the while loop can run forever. Why? Because, the calculation of
                // current = (low + high) / 2 is rounded down. (Do not set "low = current", which causes infinite loop.)
                low = current+1;
            } else if(sqr > num){
                // Decrement, because otherwise, the while loop can run forever.
                // (Do not set "high = current", which causes infinite loop.)
                high = current-1;
            } else if(sqr == num){
                return true;
            }
        }
        return false;
    }

    /**
     * Solution 2: A square number is 1+3+5+7+...
     * Running time: O(sqrt(num)).
     * @param num
     * @return
     */
    public static boolean isPerfectSquare2(int num) {
        for(int i = 1; num > 0 ; i = i+2)
        {
            num -= i;
        }
        return (num == 0);
    }

    /**
     * Solution 3: Newton's method
     * Running time: O(log(num)) on average.
     * More specifically, running time depends on how good the initial guess is. In this program, we take the initial
     * guess of x=N. Consequently there will be an initial period where x/2 is way bigger than N/x, so that the method
     * is essentially just cutting the number in half over and over. So it will take roughly log2(N/sqrt(N))=1/2log2(N)
     * steps to get to within some fixed neighborhood of sqrt(N). Then there will be a "convergence" period, which takes
     * a number of steps that really only depends on the tolerance, not on N. Adding these together will give a number
     * of steps that scales like log2(N) as N->inf, but is not directly proportional to log2(N).
     * More details 1: https://math.stackexchange.com/questions/1865688/what-is-the-computational-complexity-of-newton-raphson-method-to-find-square-roo
     * More details 2: http://en.citizendium.org/wiki/Newton%27s_method#Computational_complexity
     * @param num
     * @return
     */
    public static boolean isPerfectSquare3(int num){
        long x = num;
        while(x * x > num){
            x = (x + (num/x)) / 2;
        }
        return (x*x == num);
    }

    /**
     * Solution 4: Exponential identity
     * sqrt(num) = num^(1/2) = e^(log_e(num^(1/2))) = e ^ ((1/2)*log_e(num))
     * Details: https://en.wikipedia.org/wiki/Methods_of_computing_square_roots#Exponential_identity
     * @param num
     * @return
     */
/*
    public static boolean isPerfectSquare4(int num){
        double sqrtValue = Math.pow(10, ((1/2) * Math.log10((double)num)) );
        System.out.println(sqrtValue);
        System.out.println(sqrtValue % 1);
        System.out.println((sqrtValue % 1) == 0);
        //return Math.rint(sqrtValue) == sqrtValue;
        return (sqrtValue % 1) == 0;
    }
*/

    /**
     * Run all solutions
     * @param num
     */
    public static void runTestCases(int num){
        System.out.println("isPerfectSquare1(" + num + "): " + isPerfectSquare1(num));
        System.out.println("isPerfectSquare2(" + num + "): " + isPerfectSquare2(num));
        System.out.println("isPerfectSquare3(" + num + "): " + isPerfectSquare3(num));
        //System.out.println("isPerfectSquare4(" + num + "): " + isPerfectSquare4(num));
        System.out.println();
    }

    public static void main(String[] args){
        int number = 16;
        runTestCases(number);

        number = 5;
        runTestCases(number);

        number = 1;
        runTestCases(number);
    }
}