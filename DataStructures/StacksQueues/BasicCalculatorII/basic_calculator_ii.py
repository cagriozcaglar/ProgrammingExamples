'''
Leetcode 227: Basic Calculator II

Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, '+', '-', '*', and '/' operators.

Example 1:
Input: s = "3+2*2"
Output: 7

Example 2:
Input: s = " 3/2 "
Output: 1

Example 3:
Input: s = " 3+5 / 2 "
Output: 5
'''
'''
Solution 2: Not using stack, just last two numbers
 - Clarification: Order of parts: [prev_num] [op] [cur_num] [ch]
   - [prev_num] is at the top of the stack, can be retrieved using stack.pop()
   - [op] is the previous / last operator we have seen (not current one)
   - [cur_num] is the number we are currently building
   - [ch] is the current character we are processing. If ch is an operator, we
     evaluate the previous [prev_num] [op] [cur_num] expression.
 - Time complexity: O(n)
 - Space complexity: O(n)
'''
class Solution:
  def calculate(self, s: str) -> int:
    ans = 0
    prevNum = 0
    currNum = 0
    op = '+'

    for i, c in enumerate(s):
      if c.isdigit():
        currNum = currNum * 10 + int(c)
      if not c.isdigit() and c != ' ' or i == len(s) - 1:
        if op == '+' or op == '-':
          ans += prevNum
          prevNum = (currNum if op == '+' else -currNum)
        elif op == '*':
          prevNum *= currNum
        elif op == '/':
          prevNum = int(prevNum / currNum)
        op = c
        currNum = 0

    return ans + prevNum

'''
Solution 2: Using Stack
Somehow fails some test cases
'''
# class Solution:
#     def calculate(self, s: str) -> int:
#         stack = []
#         # Set the initial sign to '+' (plus), for the first number.
#         sign = '+'
#         # s = s.replace(" ", ""). # This results in incorrect output, commented out. Why?
#         cur_num = 0

#         for index, ch in enumerate(s):
#             if ch.isdigit():
#                 cur_num = cur_num * 10 + int(ch)
#             if (not ch.isdigit() and ch != ' ') or index == len(s) - 1:
#                 if sign == "+":
#                     stack.append(cur_num)
#                 elif sign == "-":
#                     stack.append(-cur_num)
#                 elif sign == "*":
#                     stack.append(stack.pop() * cur_num)
#                 elif sign == "/":
#                     stack.append(stack.pop() // cur_num)
#                 # Update the sign to the current operator.
#                 sign = ch
#                 cur_num = 0

#         return sum(stack)
