"""
Remove Invalid Parentheses
(Hard) (Backtracking)
"""
from typing import List

class Solution:
    def removeInvalidParentheses(self, s: str) -> List[str]:
        left = right = 0

        # Find number of misplaced left and right parentheses
        for char in s:
            if char == "(":
                left += 1
            elif char == ")":
                # If there is no matching left, this is a misplaced right
                if left == 0:
                    right += 1
                # Decrement count of left parentheses because we found right
                # parenthesis which can be a matching one for a left
                left = max(0, left-1)

        result = {}
        def backtrack(s, index, leftCount, rightCount, leftRem, rightRem, expr):
            # B1. is_a_solution: We reached end of the string, check if resulting expression is valid,
            # and also if we have removed total number of left&right
            if index == len(s):
                if leftRem == 0 and rightRem == 0:
                    # B2. process_solution
                    answer = "".join(expr)
                    result[answer] = 1
            else:
                # B3. Construct_candidates
                # 1. Discard case. Pruning condition. Don't recurse if the remaining count for that parenthesis is 0
                if (s[index] == "(" and leftRem > 0) or (s[index] == ")" and rightRem > 0):
                    backtrack(s, index+1, leftCount, rightCount, leftRem - (s[index] == '('), rightRem - (s[index] == ")"), expr)
                expr.append(s[index])

                # 2. Current character is not a parentheses, backtrack
                if s[index] not in "()":
                    backtrack(s, index+1, leftCount, rightCount, leftRem, rightRem, expr)
                # 3. Left parenthesis
                elif s[index] == "(":
                    backtrack(s, index+1, leftCount+1, rightCount, leftRem, rightRem, expr)
                # 4. Right parenthesis, with conditin leftCount > rightCount
                elif s[index] == ")" and leftCount > rightCount:
                    backtrack(s, index+1, leftCount, rightCount+1, leftRem, rightRem, expr)

                # B6. Unmake move: Pop for backtracking
                expr.pop()

        backtrack(s, 0, 0, 0, left, right, [])
        return list(result.keys())