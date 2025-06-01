'''
Leetcode 670: Maximum Swap
You are given an integer num. You can swap two digits at most once to get the maximum valued number.
Return the maximum valued number you can get.

Example 1:
Input: num = 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.

Example 2:
Input: num = 9973
Output: 9973
Explanation: No swap.
'''
class Solution:
    def maximumSwap(self, num: int) -> int:
        num_str = list(str(num))
        n = len(num_str)
        # Index of the largest digit from the current position to the end of the string
        max_right_index = [0] * n

        # Walk from right to left
        # Populate max_right_index with index of max value seen on the RHS of index
        max_right_index[n-1] = n-1
        for i in range(n-2, -1, -1):
            max_right_index[i] = i if num_str[i] > num_str[max_right_index[i+1]] \
            else max_right_index[i+1]

        # Walk from left to right
        # The first time we find a digit that is smaller than the largest one that comes
        # after it, we swap them. Since we’re always looking for the largest possible swap,
        # this guarantees that we’ll maximize the number.
        for i in range(n):
            if num_str[i] < num_str[max_right_index[i]]:
                # Swap num_str[i] and num_str[max_right_index[i]]
                num_str[i], num_str[max_right_index[i]] = num_str[max_right_index[i]], num_str[i]
                # Early return, greedy approach: result found
                return int("".join(num_str))

        # No swap found in left->right walk
        return num
