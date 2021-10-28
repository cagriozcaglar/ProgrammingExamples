"""
Note: This is a Leetcode question: https://leetcode.com/problems/1-bit-and-2-bit-characters/

We have two special characters. The first character can be represented by one bit 0. The second character can be
represented by two bits (10 or 11).

Now given a string represented by several bits. Return whether the last character must be a one-bit character or not.
The given string will always end with a zero.

Example 1:
Input:
bits = [1, 0, 0]
Output: True
Explanation:
The only way to decode it is two-bit character and one-bit character. So the last character is one-bit character.

Example 2:
Input:
bits = [1, 1, 1, 0]
Output: False
Explanation:
The only way to decode it is two-bit character and two-bit character. So the last character is NOT one-bit character.

Note:
1 <= len(bits) <= 1000.
bits[i] is always 0 or 1.

Hint:
Keep track of where the next character starts. At the end, you want to know if you started on the last bit.
"""

from typing import List


class OneBitAndTwoBitCharacter:
    @staticmethod
    def isOneBitCharacter(bits: List[int]) -> bool:
        """
        Idea:
            Allowed characters: 0, 10, 11
            Observation: Given the starting position of the character, it is easy to determine whether it is a 1-bit or 2-bit character.
                - 0 : 1-bit
                - 1 : 2-bits
            Start from left index, move to right. If the index of last character is last index of bits array, then there
            is 1-bit character at the end. Otherwise, 2-bit character, return false.
        :return: Boolean indicating whether the last character is 1-bit or 2-bits.
        """
        index, startingIndex = 0, 0
        maxIndex = len(bits)-1

        while index <= maxIndex:
            if startingIndex == maxIndex:
                return True
            # Index increment is 1 or 2, depending on whether the starting index is 0 or 1
            indexIncrement = 1 if(bits[index] == 0) else 2
            index += indexIncrement
            """
            Long version:
            # # 1-bit character
            # if bits[index] == 0:
            #     index += 1
            # # 2-bit character
            # elif bits[index] == 1:
            #     index += 2
            """
            startingIndex = index
        # If startingIndex was never equal to maxIndex, startingIndex must be lower
        # than maxIndex, and hence last character must be 2-bit character
        return False


if __name__ == '__main__':
    """
    Example 1:
    Input:
    bits = [1, 0, 0]
    Output: True
    Explanation:
    The only way to decode it is two-bit character and one-bit character. So the last character is one-bit character.
    """
    bits = [1, 0, 0]
    print(f"Is the last character of {bits} 1-bit?: {OneBitAndTwoBitCharacter.isOneBitCharacter(bits)}")

    """
    Example 2:
    Input:
    bits = [1, 1, 1, 0]
    Output: False
    Explanation:
    The only way to decode it is two-bit character and two-bit character. So the last character is NOT one-bit character.
    """
    bits = [1, 1, 1, 0]
    print(f"Is the last character of {bits} 1-bit?: {OneBitAndTwoBitCharacter.isOneBitCharacter(bits)}")
