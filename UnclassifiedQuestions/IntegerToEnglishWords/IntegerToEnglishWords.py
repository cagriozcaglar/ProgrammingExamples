"""
273. Integer to English Words
Convert a non-negative integer num to its English words representation.
"""

class Solution:
    def numberToWords(self, num: int) -> str:
        onesMap = {
            1: 'One',
            2: 'Two',
            3: 'Three',
            4: 'Four',
            5: 'Five',
            6: 'Six',
            7: 'Seven',
            8: 'Eight',
            9: 'Nine'
        }
        tenToNineteenMap = {
            10: 'Ten',
            11: 'Eleven',
            12: 'Twelve',
            13: 'Thirteen',
            14: 'Fourteen',
            15: 'Fifteen',
            16: 'Sixteen',
            17: 'Seventeen',
            18: 'Eighteen',
            19: 'Nineteen'
        }
        tensMap = {
            2: 'Twenty',
            3: 'Thirty',
            4: 'Forty',
            5: 'Fifty',
            6: 'Sixty',
            7: 'Seventy',
            8: 'Eighty',
            9: 'Ninety'
        }

        def twoDigitToWords(num):
            if not num:
                return ""
            elif num < 10:
                return onesMap[num]
            elif num < 20:
                return tenToNineteenMap[num]
            else:
                tensPart = num // 10
                rest = num - tensPart * 10
                return tensMap[tensPart] + " " + onesMap[rest] if rest else tensMap[tensPart]

        def threeDigitToWords(num):
            hundred = num // 100
            rest = num - hundred * 100
            if hundred and rest:
                return onesMap[hundred] + " Hundred " + twoDigitToWords(rest)
            elif not hundred and rest:
                return twoDigitToWords(rest)
            elif hundred and not rest:
                return onesMap[hundred] + " Hundred"

        # MAIN STARTS HERE
        billion = num // 10**9
        million = (num - billion * 10**9) // 10**6
        thousand = (num - billion * 10**9 - million * 10**6) // 10**3
        rest = (num - billion * 10**9 - million * 10**6 - thousand * 10**3)
        if not num:
            return "Zero"

        result = ""
        if billion:
            result = threeDigitToWords(billion) + " Billion"
        if million:
            result += " " if result else ""
            result += threeDigitToWords(million) + " Million"
        if thousand:
            result += " " if result else ""
            result += threeDigitToWords(thousand) + " Thousand"
        if rest:
            result += " " if result else ""
            result += threeDigitToWords(rest)
        return result