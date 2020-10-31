/**
 You're now a baseball game point recorder. Given a list of strings, each string can be one of the 4 following types:
 1. Integer (one round's score): Directly represents the number of points you get in this round.
 2. "+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
 3. "D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
 4. "C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
 Each round's operation is permanent and could have an impact on the round before and the round after.
 You need to return the sum of the points you could get in all the rounds.

 Example 1:
 Input: ["5","2","C","D","+"]
 Output: 30
 Explanation:
 Round 1: You could get 5 points. The sum is: 5.
 Round 2: You could get 2 points. The sum is: 7.
 Operation 1: The round 2's data was invalid. The sum is: 5.
 Round 3: You could get 10 points (the round 2's data has been removed). The sum is: 15.
 Round 4: You could get 5 + 10 = 15 points. The sum is: 30.

 Example 2:
 Input: ["5","-2","4","C","D","9","+","+"]
 Output: 27
 Explanation:
 Round 1: You could get 5 points. The sum is: 5.
 Round 2: You could get -2 points. The sum is: 3.
 Round 3: You could get 4 points. The sum is: 7.
 Operation 1: The round 3's data is invalid. The sum is: 3.
 Round 4: You could get -4 points (the round 3's data has been removed). The sum is: -1.
 Round 5: You could get 9 points. The sum is: 8.
 Round 6: You could get -4 + 9 = 5 points. The sum is 13.
 Round 7: You could get 9 + 5 = 14 points. The sum is 27.

 Note:
 1. The size of the input list will be between 1 and 1000.
 2. Every integer represented in the list will be between -30000 and 30000.
 */

import java.util.*;

class BaseballGame {
    /**
     * Track the baseball game score.
     * The proper data structure to use in this problem is stack, because a score at step s depends on scores at step
     * (s-2), (s-1). Using this idea, we follow the rules of baseball game listed in question description:
     * 1. Integer (one round's score): Directly represents the number of points you get in this round.
     * 2. "+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
     * 3. "D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
     * 4. "C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
     * @param ops: A string array of valid operations listed above
     * @return sum: Sum of points in all rounds
     */
    public static int calPoints(String[] ops) {
        // Use Stack. Requires "import java.util.*;"
        Stack<Integer> scores = new Stack<Integer>();
        int sum = 0;

        for(String op: ops){
            // 1. Integer (one round's score): Directly represents the number of points you get in this round.
            if(isStringAnInteger(op)){ // Uses the method written below
                int newIntegerValue = Integer.parseInt(op);
                scores.push( newIntegerValue );
                sum += newIntegerValue;
            } // 2. "+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
            else if(op.equals("+")){
                int topValue = scores.pop();
                int newTopValue = scores.peek() + topValue;
                scores.push(topValue);
                scores.push(newTopValue);
                sum += newTopValue;
            } // 3. "D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
            else if(op.equals("D")){
                scores.push( 2 * scores.peek());
                sum += scores.peek(); // Not 2x, because the number 2x is pushed to stack.
            } // 4. "C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
            else if(op.equals("C")){
                sum -= scores.pop();
            }
            // Output for debugging
            System.out.print(op + "=>" + sum + ", ");
        }

        return sum;
    }

    /**
     * Check if a string is an integer.
     * Note: The integer can be negative.
     * This method checks if a string follows the regex of [-]* (for negative integers), followed by digits [0-9]+
     * @param word
     * @return
     */
    public static boolean isStringAnInteger(String word){
        return word.matches("[-]*[0-9]+");
    }

    public static void main(String[] args){
        // Test 1
        String[] operations = {"5", "2", "C", "D", "+"};
        System.out.println(calPoints(operations));
        // Running sum values: 5=>5, 2=>7, C=>5, D=>15, +=>30
        // Output: 30

        // Test 2
        String[] operations2 = {"5", "-2", "4", "C", "D", "9", "+", "+"};
        System.out.println(calPoints(operations2));
        // Running sum values: 5=>5, -2=>3, 4=>7, C=>3, D=>-1, 9=>8, +=>13, +=>27
        // Output: 27
    }
}