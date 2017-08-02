/**
 A bracket is considered to be any one of the following characters: (, ), {, }, [, or ].
 Two brackets are considered to be a matched pair if the an opening bracket (i.e., (, [, or {) occurs to the left of a closing bracket (i.e., ), ], or }) of the exact same type. There are three types of matched pairs of brackets: [], {}, and ().
 A matching pair of brackets is not balanced if the set of brackets it encloses are not matched. For example, {[(])} is not balanced because the contents in between { and } are not balanced. The pair of square brackets encloses a single, unbalanced opening bracket, (, and the pair of parentheses encloses a single, unbalanced closing square bracket, ].
 By this logic, we say a sequence of brackets is considered to be balanced if the following conditions are met:
 1) It contains no unmatched brackets.
 2) The subset of brackets enclosed within the confines of a matched pair of brackets is also a matched pair of brackets.

 Given n strings of brackets, determine whether each sequence of brackets is balanced. If a string is balanced, print YES on a new line; otherwise, print NO on a new line.

 Input Format:
 The first line contains a single integer, , denoting the number of strings.
 Each line  of the  subsequent lines consists of a single string, , denoting a sequence of brackets.

 Constraints
 - 1 <= n <= 10^3, where n is the length of the sequence.
 - Each character in the sequence will be a bracket (i.e., {, }, (, ), [, and ]).

 Output Format:
 For each string, print whether or not the string of brackets is balanced on a new line. If the brackets are balanced, print YES; otherwise, print NO.

 ***
 Sample Input:
 3
 {[()]}
 {[(])}
 {{[[(())]]}}

 Sample Output
 YES
 NO
 YES

 Explanation:
 The string {[()]} meets both criteria for being a balanced string, so we print YES on a new line.
 The string {[(])} is not balanced, because the brackets enclosed by the matched pairs [(] and (]) are not balanced.
 The string {{[[(())]]}} meets both criteria for being a balanced string, so we print YES on a new line.

 **
 * Algorithm followed: https://stackoverflow.com/questions/23187539/java-balanced-expressions-check
 */

import java.io.*;
import java.util.*;
import java.text.*;
import java.math.*;
import java.util.regex.*;

public class BalancedBrackets {

    static Character[] leftParantheses = {'(', '{', '['};
    static Character[] rightParantheses = {')', '}', ']'};
    static HashMap<Character, Character> parMap = new HashMap<Character, Character>();

    // Initialize the parantheses map
    public static void initMap() {
        for(int i = 0; i < leftParantheses.length; i++){
            parMap.put(rightParantheses[i], leftParantheses[i]);
        }
    }

    public static String isBalanced(String s) {
        // Initialize the paranthese map
        initMap();
        Set<Character> mapKeys = parMap.keySet();
        Set<Character> values = new HashSet<Character>(Arrays.asList(leftParantheses));

        // Print for debugging
        /*
        parMap.forEach((k,v)->{
    	    System.out.println("Item : " + k + " Count : " + v);
        });
        mapKeys.forEach((k)->{
    	    System.out.println("Item : " + k);
        });
        */

        Stack<Character> stringStack = new Stack<Character>();
        for(int i = 0; i < s.length(); i++){
            Character value = s.charAt(i);
            // If character is a left paranthesis, push to stack
            if(values.contains(value)){
                stringStack.push(value);
            }
            // Else if character is a right paranthesis
            else if(mapKeys.contains(value)){
                // If stack is empty, not matching.
                if(stringStack.empty()){
                    return "NO";
                }
                // If value of top of stack is not the matching pair of value, not matching.
                if(parMap.get(value) != (stringStack.pop())){
                    return "NO";
                }
            }
        }
        // If stack is not empty, not matching. If stack is empty, matching.
        return (stringStack.empty()) ? "YES" : "NO";
    }

    public static void main(String[] args) {
        Scanner in = new Scanner(System.in);
        int t = in.nextInt();
        for(int a0 = 0; a0 < t; a0++){
            String s = in.next();
            String result = isBalanced(s);
            System.out.println(result);
        }
        in.close();
    }
}