'''
Leetcode 2232: Minimize Result by Adding Parentheses to Expression

You are given a 0-indexed string expression of the form "<num1>+<num2>" where <num1> and <num2> represent positive integers.

Add a pair of parentheses to expression such that after the addition of parentheses, expression is a valid mathematical expression and evaluates to the smallest possible value. The left parenthesis must be added to the left of '+' and the right parenthesis must be added to the right of '+'.

Return expression after adding a pair of parentheses such that expression evaluates to the smallest possible value. If there are multiple answers that yield the same result, return any of them.

The input has been generated such that the original value of expression, and the value of expression after adding any pair of parentheses that meets the requirements fits within a signed 32-bit integer.

Example 1:
Input: expression = "247+38"
Output: "2(47+38)"
Explanation: The expression evaluates to 2 * (47 + 38) = 2 * 85 = 170.
Note that "2(4)7+38" is invalid because the right parenthesis must be to the right of the '+'.

Example 2:
Input: expression = "12+34"
Output: "1(2+3)4"
Explanation: The expression evaluates to 1 * (2 + 3) * 4 = 1 * 5 * 4 = 20.
'''
import math

class Solution:
    def minimizeResult(self, expression: str) -> str:
        min_value = math.inf
        selected_expr = ""
        # Find the index of + sign
        plus_index = expression.find("+")
        # Left and right substrings
        left = expression[:plus_index]
        right = expression[plus_index+1:]

        for left_index in range(len(left)):
            for right_index in range(len(right)):
                # 1. Four parts of the expression
                before_left_paren = 1 if left_index == 0 else int(left[:left_index])
                first_term = int(left[left_index:])
                second_term = int(right[0: right_index+1])
                after_right_paren = 1 if right_index == len(right)-1 else int(right[right_index+1:])
                # 2. Calculate the value of expression using four parts
                new_value = before_left_paren * (first_term + second_term) * after_right_paren

                # 3. Update min value, update selected expression
                # Note: Do not update selected expression, if min value didn't change.
                if min_value > new_value:
                    min_value = new_value
                    selected_expr = ('' if left_index == 0 else str(before_left_paren)) + \
                    "(" + \
                    str(first_term) + \
                    "+" + \
                    str(second_term) + \
                    ")" + \
                    ('' if right_index == len(right)-1 else str(after_right_paren))


        return selected_expr
