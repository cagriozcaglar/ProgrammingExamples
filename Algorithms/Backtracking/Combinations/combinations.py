"""
Leetcode 77: Combinations
Given two integers n and k, return all possible combinations of k
numbers out of the range [1, n].
"""
from typing import List

class Solution:    
    def combineEPI(self, n: int, k: int) -> List[List[int]]:
        def directed_combs(offset, partial_comb):
            # Base case: length of partial_comb is k, add subset
            if len(partial_comb) == k:
                result.append(partial_comb)
                return  # IMPORTANT: Do not forget to return

            # If size is not k yet, generate remaining combs over
            # {offset,..,n-1} of size num_remaining
            num_remaining = k - len(partial_comb)
            i = offset
            while i <= n and num_remaining <= n - i + 1:
                directed_combs(i + 1, partial_comb + [i])
                i += 1

        result = []
        directed_combs(1, [])  # Why 1?
        return result
    
    def combineLeetcode(self, n: int, k: int) -> List[List[int]]:
        def backtrack(curr, first_num):
            # Base case
            if len(curr) == k:
                combs.append(curr[:])
                return

            # Need, remain, available
            need = k - len(curr)
            remain = n - first_num + 1
            available = remain - need

            for num in range(first_num, first_num + available + 1):
                # Make move
                curr.append(num)
                # Backtrack
                backtrack(curr, num + 1)
                # Unmake move
                curr.pop()

        combs = []
        backtrack([], 1)
        return combs
