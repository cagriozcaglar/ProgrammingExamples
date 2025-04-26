from typing import List

class Solution:
    # Solution 1: Backtracking, mentioned in EPI book
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def directed_subset(to_be_selected: int, selected_so_far: List[int]):
            # Base case: to_be_selected = len(nums)
            if to_be_selected == len(nums):
                subsets.append(list(selected_so_far))
                return

            # Two cases: combinations without and with nums[to_be_selected]
            # 1) Combinations without nums[to_be_selected]
            directed_subset(to_be_selected + 1, selected_so_far)
            # 2) Combinations with nums[to_be_selected]
            directed_subset(to_be_selected + 1, selected_so_far + [nums[to_be_selected]])

        subsets = []
        directed_subset(0, [])
        return subsets

    # Solution 2: Bitmasking
    def subsets_with_bitmask(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        output = []

        for i in range(2**n, 2**(n+1)):
            # Generate bitmask, from 0..00 to 1..11
            # Explanation: bin(i) returns binary representation bin(4) = '0b100'
            # bin(i)[3:] removes the first 3 characters:
            # 1) First two characters are '0b', not needed.
            # 2) Third character is "1", because the numbers start from 2**n, and
            # we do not need this character, because we are using [0, 2**n-1]
            # which uses all characters except the first 3 characters
            bitmask = bin(i)[3:]

            # Append subset corresponding to that bitmask
            output.append([nums[j] for j in range(n) if bitmask[j] == "1"])

        return output

    # Solution 3: Backtracking from Leetcode
    def subsets_backtracking(self, nums: List[int]) -> List[List[int]]:
        self.output = []
        self.n = len(nums)
        self.backtrack(0, [], nums)
        return self.output

    def backtrack(self, first, curr, nums):
        # Add the current subset to the output
        self.output.append(curr[:])
        # Generate subsets starting from the current index
        for i in range(first, self.n):
            # Make move
            curr.append(nums[i])
            # Backtrack
            self.backtrack(i+1, curr, nums)
            # Unmake move
            curr.pop()


    # Solution 4: Cascading
    def subsets_cascading(self, nums: List[int]) -> List[List[int]]:
        output = [[]]
        for num in nums:
            newSubsets = []
            for curr in output:
                temp = curr.copy()
                temp.append(num)
                newSubsets.append(temp)
            for curr in newSubsets:
                output.append(curr)
        return output







