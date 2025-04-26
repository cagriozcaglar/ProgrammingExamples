'''
Leetcode 564: Find the Closest Palindrome

Given a string n representing an integer, return the closest integer (not including itself), which is a palindrome. If there is a tie, return the smaller one.

The closest is defined as the absolute difference minimized between two integers.

Example 1:
Input: n = "123"
Output: "121"

Example 2:
Input: n = "1"
Output: "0"
'''
from typing import List

class Solution:
    def nearestPalindromic(self, n: str) -> str:
        len_n = len(n)
        num = int(n)
        i = len_n // 2 - 1 if len_n % 2 == 0 else len_n // 2
        first_half = int(n[: i+1])
        """
        Generate possible palindromic candidates:
        1. Create a palindrome by mirroring the first half.
        2. Create a palindrome by mirroring the first half incremented by 1.
        3. Create a palindrome by mirroring the first half decremented by 1.
        4. Handle edge cases by considering palindromes of the form 999... 
           and 100...001 (smallest and largest n-digit palindromes).
        """
        candidates: List[int] = []

        # Generate palindromes by varying the first_half by -1, 0, +1
        for i in range(first_half - 1, first_half + 2):
            # For even lengths, use the entire prefix.
            # For odd lengths, exclude the last digit of the prefix.
            j = i if len_n % 2 == 0 else i // 10

            # Append the reverse of 'j' to 'i' to construct the palindrome
            palindrome = i
            while j > 0:
                palindrome = palindrome * 10 + j % 10
                j //= 10

            # Add the constructed palindrome to the candidate set
            candidates.append(palindrome)


        # Initialize a set with the smallest and largest possible palindromes
        # with different digit lengths compared to the input number
        candidates.extend([
            10 ** (len_n - 1) - 1,  # Smallest palindrome with one less digit
            10 ** len_n + 1         # Smallest palindrome with one more digit
        ])

        # Original num may end up in the candidates, if so, remove it.
        if num in candidates:
            candidates.remove(num)

        # Find closest palindrome
        closest_palindrome = -1
        for candidate in candidates:
            if closest_palindrome == -1 or \
                (abs(candidate - num) < abs(closest_palindrome - num)) or \
                (abs(candidate - num) == abs(closest_palindrome - num) and candidate < closest_palindrome):
                closest_palindrome = candidate

        # Convert the closest palindrome back to a string and return
        return str(closest_palindrome)