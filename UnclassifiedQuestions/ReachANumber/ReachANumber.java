/**
 You are standing at position 0 on an infinite number line. There is a goal at position target.
 On each move, you can either go left or right. During the n-th move (starting from 1), you take n steps.
 Return the minimum number of steps required to reach the destination.

 Example 1:
 Input: target = 3
 Output: 2
 Explanation:
 On the first move we step from 0 to 1.
 On the second step we step from 1 to 3.

 Example 2:
 Input: target = 2
 Output: 3
 Explanation:
 On the first move we step from 0 to 1.
 On the second move we step  from 1 to -1.
 On the third move we step from -1 to 2.

 Note: target will be a non-zero integer in the range [-10^9, 10^9].
 **/

public class ReachANumber {
    /**
     *
     * @param target
     * @return
     */
    public static int reachNumber(int target) {
        int absoluteTarget = Math.abs(target);
        int sum = 0;
        int stepCount = 0;
        // Sum from 1 to n until sum >= absoluteTarget
        while(sum < absoluteTarget) {
            stepCount++;
            sum += stepCount;
        }

        // 1. If sum == target, use 1+2+...+n = target. Return stepCount.
        if(sum == target){
            return stepCount;
        }

        int diff = sum - target;

        // 2. If sum > target and difference is even, let k = diff/2, use 1+2+..+(k-1)-k+(k+1)+...+n = target (only sign change). Return stepCount.
        if( diff % 2 == 0){
            return stepCount;
        }
        else {
            // 3. If sum > target and difference is odd, make the difference even by proceeding as follows:
            //  a. If (stepCount+1) is odd: diff + (stepCount+1) is even, so we only need to add (stepCount+1). Since diff is even in this case, we can use the -diff/2 as in step 2. Return (stepCount+1).
            if((stepCount+1) % 2 == 1) {
                return stepCount+1;
            }
            // b. If (stepCount+1) is even: diff + (stepCount+1) + (stepCount+2) is even, so we need to add (stepCount+1) + (stepCount+2). Since diff is even in this case, we can use the -diff/2 as in step 2. Return (stepCount+2).
            else {
                return stepCount+2;
            }
        }
    }

    public static void runTestCase(int target){
        System.out.println("Number of minimum steps to reach " + target + ": " + reachNumber(target));
    }

    public static void main(String[] args){
        int target = 15;
        runTestCase(target);

        target = 2;
        runTestCase(target);

        target = 3;
        runTestCase(target);

        target = 4;
        runTestCase(target);

        target = 5;
        runTestCase(target);

        target = -5;
        runTestCase(target);

        target = 1000000000;
        runTestCase(target);

        target = -1000000000;
        runTestCase(target);
    }
}