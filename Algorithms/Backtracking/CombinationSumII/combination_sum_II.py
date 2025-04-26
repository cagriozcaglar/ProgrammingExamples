'''
Leetcode 40. Combination Sum II

Given a collection of candidate numbers (candidates) and a target number (target),
find all unique combinations in candidates where the candidate numbers sum to target.

Each number in candidates may only be used once in the combination.

Note: The solution set must not contain duplicate combinations.
'''

# Time complexity: O(2^n)
# Space complexity: O(N)
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        def gen_combs(comb: List[List[int]], tempList: List[int], candidates: List[int], remaining: int, startIndex: int) -> None:
            if remaining < 0:
                return
            # Base case
            elif remaining == 0:
                results.append(tempList.copy())
                return
            # Generate candidates: make move, backtrack, unmake move
            else:
                for i in range(startIndex, len(candidates)):
                    # Consecutive duplicate record check, we don't want dup combs
                    if i > startIndex and candidates[i] == candidates[i-1]:
                        continue
                    # If remaining is lower than next candidate, break
                    if remaining < candidates[i]:
                        break
                    # Make move: Add candidates[i] to tempList
                    tempList.append(candidates[i])
                    # Backtrack
                    gen_combs(
                        comb=comb,
                        tempList=tempList,
                        candidates=candidates,
                        remaining=remaining - candidates[i],
                        startIndex=i + 1
                    )
                    # Unmake move
                    tempList.pop()

        results = []
        candidates.sort()
        gen_combs(
            comb=results,
            tempList=[],
            candidates=candidates,
            remaining=target,
            startIndex=0
        )
        return results
