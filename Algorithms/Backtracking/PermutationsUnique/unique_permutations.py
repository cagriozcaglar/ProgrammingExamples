from collections import Counter
from typing import List

class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        def backtrack(perm, counter):
            # Base case: permutation found, length reached
            if len(perm) == len(nums):
                # Make a deep copy of perm, because perm will be backtracked later.
                results.append(list(perm))
                return

            for num in counter:  # Big difference: iterate over unique values
                if counter[num] > 0:
                    # Make move: Add number to current perm
                    perm.append(num)
                    counter[num] -= 1  # Decrement counter for this num
                    # Backtrack
                    backtrack(perm, counter)
                    # Unmake move: Remove number from current perm
                    perm.pop()
                    counter[num] += 1  # Increment counter for this num

        results = []
        backtrack([], Counter(nums))
        return results
    
def __main__():
    print(Solution().permuteUnique([1,1,2]))
    # [[1, 1, 2], [1, 2, 1], [2, 1, 1]]

if __name__ == "__main__":
    __main__()