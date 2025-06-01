'''
Leetcode 282: Expression Add Operators

Given a string num that contains only digits and an integer target, return all possibilities to insert the binary operators '+', '-', and/or '*' between the digits of num so that the resultant expression evaluates to the target value.
Note that operands in the returned expressions should not contain leading zeros.

Example 1:
Input: num = "123", target = 6
Output: ["1*2*3","1+2+3"]
Explanation: Both "1*2*3" and "1+2+3" evaluate to 6.

Example 2:
Input: num = "232", target = 8
Output: ["2*3+2","2+3*2"]
Explanation: Both "2*3+2" and "2+3*2" evaluate to 8.

Example 3:
Input: num = "3456237490", target = 9191
Output: []
Explanation: There are no expressions that can be created from "3456237490" to evaluate to 9191.
'''
from typing import List

'''
- Time Complexity: O(N×4^N)
  - At every step along the way, we consider exactly 4 different choices or 4 different recursive paths. The base case is when the value of index reaches N i.e. the length of the nums array. Hence, our complexity would be O(4^N).
  - For the base case we use a StringBuilder::toString operation in Java and .join() operation in Python and that takes O(N) time. Here N represents the length of our expression. In the worst case, each digit would be an operand and we would have N digits and N−1 operators. So O(N). This is for one expression. In the worst case, we can have O(4^N) valid expressions.
  - Overall time complexity = O(N×4^N).

- Space Complexity: O(N)
  - For both Python and Java implementations we have a list data structure that we update on the fly and only for valid expressions do we create a new string and add to our answers array. So, the space occupied by the intermediate list would be O(N) since in the worst case the expression would be built out of all the digits as operands.
  - Additionally, the space used up by the recursion stack would also be O(N) since the size of recursion stack is determined by the value of index and it goes from 0 all the way to N.
'''
class Solution:
    def addOperators(self, num: str, target: int) -> List[str]:
        n = len(num)
        answers = []

        def generate_expressions(
            index: int,
            prev_operand: int,
            current_operand: int,
            value: int,
            string: str,
        ) -> None:
            # Done processing all digits in num
            if index == n:
                # If final value == target and no operand is left unprocessed
                if value == target and current_operand == 0:
                    answers.append("".join(string[1:]))
                return

            # Extending the current operand by one digit
            current_operand = current_operand * 10 + int(num[index])
            str_op = str(current_operand)

            # CASE 1: NO OPERATOR, JUST CONTACT DIGITS
            # To avoid cases where 1+05 or 1*05, because 05 won't be
            # a valid operand. Hence, check if current_operand > 0.
            if current_operand > 0:
                # no-op recursion
                generate_expressions(index+1, prev_operand, current_operand, value, string)

            # CASE 2: ADDITION
            string.append('+')
            string.append(str_op)
            generate_expressions(index+1, current_operand, 0, value + current_operand, string)
            string.pop()
            string.pop()  # Pop twice: second pop is for + sign added above

            # Can subtract or multiply if there are some previous operands
            if string:

                # CASE 3: SUBTRACTION
                string.append('-')
                string.append(str_op)
                generate_expressions(index+1, -current_operand, 0, value - current_operand, string)
                string.pop()
                string.pop()  # Pop twice: second pop is for - sign added above

                # CASE 4: MULTIPLICATION
                string.append('*')
                string.append(str_op)
                generate_expressions(index+1, current_operand * prev_operand, 0, value - prev_operand + (current_operand * prev_operand), string)
                string.pop()
                string.pop()  # Pop twice: second pop is for * sign added above


        generate_expressions(0, 0, 0, 0, [])
        return answers