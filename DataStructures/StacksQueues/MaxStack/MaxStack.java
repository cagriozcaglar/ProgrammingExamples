/**
// HackerRank challenge: https://www.hackerrank.com/challenges/maximum-element
You have an empty sequence, and you will be given N queries. Each query is one of these three types:
1 x  -Push the element x into the stack.
2    -Delete the element present at the top of the stack.
3    -Print the maximum element in the stack.
Input Format
The first line of input contains an integer, N. The next N lines each contain an above mentioned query. (It is guaranteed that each query is valid.)
Constraints:
- 1 <= N <= 10^5
- 1 <= x <= 10^9
- 1 < type < 3

Output Format
For each type 3 query, print the maximum element in the stack on a new line.
Sample Input:
10
1 97
2
1 20
2
1 26
1 20
2
3
1 91
3
Sample Output:
26
91
**/

import java.io.*;
import java.util.*;

class StackWithMax extends Stack<Integer> {
    private Stack<Integer> maxStack;
    public StackWithMax () {
        maxStack = new Stack<Integer>();
    }
    // Push element
    public void push(int value){
        if(maxStack.size() > 0 && value >= max()){
            maxStack.push(value);
        }
        if(maxStack.size() == 0){
            maxStack.push(value);
        }
        super.push(value);
    }

    // Delete element
    public Integer pop(){
        Integer value = super.pop();
        if(maxStack.size() > 0 && value >= max()){
            maxStack.pop();
        }
        return value;
    }

    // Get maximum
    public Integer max(){
        return maxStack.peek();
    }
}

public class MaxStack {
    public static void main(String[] args) {
        /* Enter your code here. Read input from STDIN. Print output to STDOUT. Your class should be named Solution. */
        Scanner scanner = new Scanner(System.in);

        // Empty Stack
        StackWithMax stack = new StackWithMax();

        // Get number of instructions
        String[] instructions = new String[scanner.nextInt()];
        scanner.nextLine();

        System.out.println("\n" + "Output:");
        for(int i = 0; i < instructions.length; i++){
            instructions[i] = scanner.nextLine();
            String[] instructionArray = instructions[i].trim().split(" ");
            // Push element, update max
            if(instructionArray[0].equals("1")){
                stack.push(Integer.parseInt(instructionArray[1]));
            }
            // Pop element
            else if(instructionArray[0].equals("2")){
                stack.pop();
            }
            // Print max
            else if(instructionArray[0].equals("3")){
                if(stack.max() != null){
                    System.out.println(stack.max());
                }
            }
        }
    }
}