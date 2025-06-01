'''
Leetcode 22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example 1:
Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]

Example 2:
Input: n = 1
Output: ["()"]
'''
from typing import List
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        parens: List[str] = []

        def gen_parens(l_count: int, r_count: int, parens_so_far: str) -> None:
            # IMPORTANT: This error condition needs to be checked first, before the next positive condition
            # This is to prevent adding bad outputs to the result.
            if r_count > l_count or l_count > n or r_count > n:
                return
            if len(parens_so_far) == 2 * n:
                parens.append(parens_so_far)
                return
            # Make move: left (
            gen_parens(l_count + 1, r_count, parens_so_far + "(")
            # Make move: rigth )
            gen_parens(l_count, r_count + 1, parens_so_far + ")")

        gen_parens(
            l_count=0,
            r_count=0,
            parens_so_far=""
        )
        return parens
