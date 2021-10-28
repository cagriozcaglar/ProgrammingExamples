"""
39. Combination Sum
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of
candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency
of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the
given input.
"""
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        results = []

        def backtrack(remain, comb, start):
            # 1. Is_a_solution
            # 2. Process solution
            if remain == 0:
                # Make a deep copy of the current combo
                results.append(list(comb))
                return
            elif remain < 0:
                # Exceeded the scope, stop exploration
                return
            # 3. Generate candidates
            for i in range(start, len(candidates)):
                # 4. Make move: Add number to combo
                comb.append(candidates[i])
                # 5. Backtrack: Try current number again
                backtrack(remain-candidates[i], comb, i)
                # 6. Backtrack, remove number from combo
                comb.pop()

        backtrack(target, [], 0)
        return results