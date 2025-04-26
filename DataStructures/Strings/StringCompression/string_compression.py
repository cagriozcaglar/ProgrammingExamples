'''
Leetcode 443: String Compression

Given an array of characters chars, compress it using the following algorithm:

Begin with an empty string s. For each group of consecutive repeating characters in chars:

If the group's length is 1, append the character to s.
Otherwise, append the character followed by the group's length.

The compressed string s should not be returned separately, but instead be stored in the input character array chars. Note that group lengths that are 10 or longer will be split into multiple characters in chars.

After you are done modifying the input array, return the integer length of the array.

Example 1:
Input: chars = ["a","a","b","b","c","c","c"]
Output: Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".

Example 2:
Input: chars = ["a"]
Output: Return 1, and the first character of the input array should be: ["a"]
Explanation: The only group is "a", which remains uncompressed since it's a single character.

Example 3:
Input: chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
Output: Return 4, and the first 4 characters of the input array should be: ["a","b","1","2"].
Explanation: The groups are "a" and "bbbbbbbbbbbb". This compresses to "ab12".
'''

from typing import List

class Solution:
    def compress(self, chars: List[str]) -> int:
        # Pointers
        read, write, length = 0, 0, len(chars)

        # Process char list
        while read < length:
            read_next = read + 1
            # Same char
            while read_next < length and chars[read_next] == chars[read]:
                read_next += 1

            # Write current char
            chars[write] = chars[read]
            write += 1

            # Write count, based on length
            if read_next - read > 1:
                count = str(read_next - read)
                chars[write: write + len(count)] = list(count)
                write += len(count)

            # Move read pointer to next new character
            read = read_next

        # Return length, which is equal to write
        return write

    def compressMyTerribleAttempt(self, chars: List[str]) -> int:
        if len(chars) == 0:
            return 0
        comp_length = 0
        pointer = 0
        count = 1
        for i in range(1, len(chars)):
            if chars[i] == chars[i - 1]:
                count += 1
            # Else, write, reset count
            else:
                # length of char, plus length of count
                count_length = len(str(count))
                comp_length += 1 + count_length if count > 1 else 1
                chars[pointer] = chars[i - 1]
                pointer += 1
                if count > 1:
                    chars[pointer: pointer + count_length] = list(str(count))
                    pointer += count_length
                # Updates for next iteration
                count = 1
                # pointer = pointer + comp_length
        # Edge case: last character is added to comp_length
        # length of char, plus length of count
        count_length = len(str(count))
        chars[pointer] = chars[-1]
        pointer += 1
        if count > 1:
            chars[pointer: pointer + count_length] = list(str(count))
            pointer += count_length

        return pointer
