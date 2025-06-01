'''
LeetCode 20. Valid Parentheses

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
Every close bracket has a corresponding open bracket of the same type.
'''
from collections import deque

class Solution:
    def isValid(self, s: str) -> bool:
        stack = deque([])
        # .append() for push() (push to front)
        # .pop() for pop() (pop from front)
        opening_brackets = ['{', '(', '[']
        closing_brackets = ['}', ')', ']']
        close_to_open_map = dict(zip(closing_brackets, opening_brackets))

        for paren in s:
            # print(f'Processing {paren}')
            if paren in opening_brackets:
                stack.append(paren)
            elif paren in closing_brackets:
                if len(stack) >= 1 and stack[-1] != close_to_open_map[paren]:
                    return False
                # If stack is empty, we cannot pop. Return false
                if not stack:
                    return False
                stack.pop()
            else:
                return False

        return len(stack) == 0