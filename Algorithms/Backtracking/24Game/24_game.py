'''
Leetcode 679: 24 Game

You are given an integer array cards of length 4. You have four cards, each containing a number in the range [1, 9]. You should arrange the numbers on these cards in a mathematical expression using the operators ['+', '-', '*', '/'] and the parentheses '(' and ')' to get the value 24.

You are restricted with the following rules:

- The division operator '/' represents real division, not integer division.
  For example, 4 / (1 - 2 / 3) = 4 / (1 / 3) = 12.
- Every operation done is between two numbers. In particular, we cannot use '-' as a unary operator.
  For example, if cards = [1, 1, 1, 1], the expression "-1 - 1 - 1 - 1" is not allowed.
- You cannot concatenate numbers together
  For example, if cards = [1, 2, 1, 2], the expression "12 + 12" is not valid.

Return true if you can get such expression that evaluates to 24, and false otherwise.

Example 1:
Input: cards = [4,1,8,7]
Output: true
Explanation: (8-4) * (7-1) = 24

Example 2:
Input: cards = [1,2,1,2]
Output: false
'''
from typing import List

class Solution:
    # All possible operations we can perform on two numbers.
    def generate_possible_results(self, a: float, b: float) -> List[float]:
        res = [a + b, a - b, b - a, a * b]
        if a:
            res.append(b / a)
        if b:
            res.append(a / b)  
        return res
    
    # Check if using current list we can react result 24.
    def check_if_result_reached(self, cards: List[float]) -> bool:
        # Base Case: We have only one number left, check if it is approximately 24.
        if len(cards) == 1:
            return abs(cards[0] - 24.0) <= 0.1

        for i in range(len(cards)):
            for j in range(i + 1, len(cards)):
                # Create a new list with the remaining numbers and the new result.
                new_list = [number for k, number in enumerate(cards) if (k != i and k != j)]
                
                # For any two numbers in our list, we perform every operation one by one.
                for res in self.generate_possible_results(cards[i], cards[j]):
                    # Add the new result to the list.
                    new_list.append(res)
                    
                    # Check if using this new list we can obtain the result 24.
                    if self.check_if_result_reached(new_list):
                        return True
                    
                    # Backtrack: remove the result from the list.
                    new_list.pop()
                    
        return False
    
    def judgePoint24(self, cards: List[int]) -> bool:
        return self.check_if_result_reached(cards)