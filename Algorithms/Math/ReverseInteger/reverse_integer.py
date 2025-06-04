'''
Leetcode 7: Reverse Integer

Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-2^31, 2^31 - 1], then return 0.
'''

class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if x < 0 else 1
        reverse, x = 0, abs(x)

        while x:
            x, mod = divmod(x, 10)
            reverse = reverse * 10 + mod
            if reverse > 2**31 - 1:
                return 0

        return sign * reverse