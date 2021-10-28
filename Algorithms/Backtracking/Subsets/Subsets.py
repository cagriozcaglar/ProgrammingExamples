"""
78. Subsets
Given an integer array nums of unique elements, return all possible subsets (the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.
"""
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = []
        for i in range(2**n, 2**(n+1)):
            # Generate bitmask, from 0..00 to 1..1
            # Note: bin(i) returns with prefix 0b (e.g. bin(5) => '0b101')
            bitmask = bin(i)[3:]
            # Append subset corresponding to the bitmask
            output.append([nums[j] for j in range(n) if bitmask[j] == '1'])
        return output

    def subsetsBacktracking(self, nums: List[int]) -> List[List[int]]:
        def backtrack(first=0, curr=[]):
            # 1. Solution found
            if len(curr) == k:
                # 2. Process solution
                output.append(curr[:])
                return
            # 3. Generate candidates
            for i in range(first, n):
                # 4. Make move: Add nums[i] to subset
                curr.append(nums[i])
                # 5. Backtrack: Use next integers to complete the subset
                backtrack(i+1, curr)
                # 6. Unmake move
                curr.pop()

        output = []
        n = len(nums)
        for k in range(n+1):
            backtrack()
        return output