"""
Return all **distinct** permutations of the given string.
For example: 
generatePermutations("abc") returns {"abc", "acb", "bac", "bca", "cab", "cba"}
generatePermutations("aaa") returns {"aaa"}
"""

from itertools import product

def permutations(inputString):
    result = set()
    def permutes(inputStr):
        for pos in range(len(inputStr)):
            leftPerms = permutes(inputStr[:pos])
            rightPerms = permutes(inputStr[pos:])
            for l, r in product(leftPerms, rightPerms):
                result |= {''.join([l, inputStr[pos], r])}
    return permutes(inputString)

if __name__ == "__main__":
    inputString = "abc"
    permutations(inputString)