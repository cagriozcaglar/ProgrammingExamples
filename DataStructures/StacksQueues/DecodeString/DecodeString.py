"""
Note: This is a Leetcode question: https://leetcode.com/problems/decode-string/

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly
k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat
numbers, k. For example, there won't be input like 3a or 2[4].

Example 1:
Input: s = "3[a]2[bc]"
Output: "aaabcbc"

Example 2:
Input: s = "3[a2[c]]"
Output: "accaccacc"

Example 3:
Input: s = "2[abc]3[cd]ef"
Output: "abcabccdcdcdef"

Example 4:
Input: s = "abc3[cd]xyz"
Output: "abccdcdcdxyz"
"""
from collections import deque


class DecodeString:
    @staticmethod
    def decodeString(s: str) -> str:
        stack = deque()
        currentNumber = 0
        currentString = ""
        for char in s:
            # If digit
            if char.isdigit():  # i >= "0" and i <= "9":
                currentNumber = 10 * currentNumber + int(char)
            # If [ => end of a number: Finalize number and string, then set them to empty
            if char == "[":
                # When you push, append string first, number last (number at top)
                stack.append(currentString)
                stack.append(currentNumber)
                currentString = ""
                currentNumber = 0
            # If ] => end of a string: Pop number and previous string
            # Update current string using number and previous string
            if char == "]":
                # When you pop, pop number first, string last (number at top)
                num = stack.pop()
                prevString = stack.pop()
                currentString = prevString + num * currentString
                currentNumber = 0
            # If character
            elif char.isalpha():
                currentString += char
        return currentString


if __name__ == "__main__":
    """
    Example 1:
    Input: s = "3[a]2[bc]"
    Output: "aaabcbc"
    """
    s = "3[a]2[bc]"
    print(f"Decoded string for \"{s}\" is: \"{DecodeString.decodeString(s)}\"")

    """    
    Example 2:
    Input: s = "3[a2[c]]"
    Output: "accaccacc"
    """
    s = "3[a2[c]]"
    print(f"Decoded string for \"{s}\" is: \"{DecodeString.decodeString(s)}\"")

    """
    Example 3:
    Input: s = "2[abc]3[cd]ef"
    Output: "abcabccdcdcdef"
    """
    s = "2[abc]3[cd]ef"
    print(f"Decoded string for \"{s}\" is: \"{DecodeString.decodeString(s)}\"")

    """
    Example 4:
    Input: s = "abc3[cd]xyz"
    Output: "abccdcdcdxyz"
    """
    s = "abc3[cd]xyz"
    print(f"Decoded string for \"{s}\" is: \"{DecodeString.decodeString(s)}\"")
