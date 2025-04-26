'''
Leetcode 17: Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example 1:
Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
'''

from typing import List

class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []

        # Map all the digits to their corresponding letters
        num_to_letter = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        # List of combinations to return
        combs = []

        def letter_comb_generator(curr_index: int = 0, letters_so_far: str = "") -> None:
            if curr_index == len(digits):
                combs.append(letters_so_far)
                return

            digit = digits[curr_index]

            for letter in num_to_letter[digit]:
                # Make move, Backtrack, Unmake move: All in one line.
                letter_comb_generator(curr_index + 1, letters_so_far + letter)

        letter_comb_generator()
        return combs
