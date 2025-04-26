'''
Leetcode 166: Fraction to Recurring Decimal

Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.

If the fractional part is repeating, enclose the repeating part in parentheses.

If multiple answers are possible, return any of them.

It is guaranteed that the length of the answer string is less than 104 for all the given inputs.

Example 1:
Input: numerator = 1, denominator = 2
Output: "0.5"

Example 2:
Input: numerator = 2, denominator = 1
Output: "2"

Example 3:
Input: numerator = 4, denominator = 333
Output: "0.(012)"
'''

from collections import defaultdict

# The solution below simulates Long Division learned in elementary school
class Solution:
    def fractionToDecimalV2MyTry(self, numerator: int, denominator: int) -> str:
        if numerator == 0:
            return "0"

        result = ""
        sign = "-" if numerator * denominator < 0 else ""
        result += sign
        
        # Get absolute values of numerator and denominator, now that we handled the sign
        numerator = abs(numerator)
        denominator = abs(denominator)
        
        # whole integer
        whole_number = numerator // denominator
        result += str(whole_number)
        
        # Remainder
        rem = numerator % denominator
        # Numerator divisible by denominator => No fraction part. Return whole number.
        if rem == 0:
            return result
        # Decimal, add "." before fractional part starts
        result += "."
        
        # Map remainder to index in result
        index_map: defaultdict(int) = {}
        
        # Divide rem, until you see the same value
        while rem != 0:
            if rem in index_map:
                result = result[0:index_map[rem]] + "(" + result[index_map[rem]:] + ")"
                return result
            index_map[rem] = len(result)
            rem = rem * 10
            val = rem // denominator
            result += str(val)
            rem = rem % denominator
        
        return result

    def fractionToDecimal(self, numerator: int, denominator: int) -> str:
        if numerator == 0:
            return "0"

        fraction = []
        # Sign: Negative or positive
        if (numerator < 0) != (denominator < 0):
            fraction.append("-")

        # Get absolute values of numerator and denominator, now that we handled the sign
        dividend = abs(numerator)
        divisor = abs(denominator)

        # Get whole number part
        fraction.append(str(dividend // divisor))
        # Get fractional part
        remainder = dividend % divisor
        # Numerator divisible by denominator => No fraction part. Return whole number.
        if remainder == 0:
            return "".join(fraction)

        # Fractional part starts here
        fraction.append(".")
        # lookup maps a digit (in fractional part) to its index in fraction variable
        lookup = {}

        while remainder != 0:
            # Found the first repeating digit. 
            if remainder in lookup:
                # Insert opening bracket ( at the position of the first repeating digit
                fraction.insert(lookup[remainder], "(")
                # Append ) at the end to close repeating digits
                fraction.append(")")
                break

            lookup[remainder] = len(fraction)
            # Multiply remainder by 10
            remainder *= 10
            # Divide remainder by divisor again, 
            fraction.append(str(remainder // divisor))
            # Update remainder
            remainder %= divisor

        return "".join(fraction)