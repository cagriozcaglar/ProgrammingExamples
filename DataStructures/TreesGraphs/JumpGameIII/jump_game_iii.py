'''
Leetcode 1306: Jump Game III

Given an array of non-negative integers arr, you are initially positioned at start index of
the array. When you are at index i, you can jump to i + arr[i] or i - arr[i], check if you
can reach to any index with value 0.

Notice that you can not jump outside of the array at any time.

Example 1:
Input: arr = [4,2,3,0,3,1,2], start = 5
Output: true
Explanation:
All possible ways to reach at index 3 with value 0 are:
1. 5 -> 6 -> 4 -> 1 -> 3
2. 5 -> 6 -> 4 -> 2 -> 0 -> 3
'''
from typing import List
from collections import deque

class Solution:
    # BFS-based.
    def canReachBfs(self, arr: List[int], start: int) -> bool:
        n = len(arr)
        q = deque([start])
        visited = set()

        while q:
            node = q.popleft()  # Check if we reached zero
            # Check if we reached zero
            if arr[node] == 0:
                return True
            if node in visited:
                continue

            # Check neighbours
            for i in [node + arr[node], node - arr[node]]:
                if 0 <= i < n:  # within bounds
                    q.append(i)

            # Mark as visited
            visited.add(node)

        return False


    # DFS-based:
    # Note: nodes are marked as visited, by setting their values to negative
    def canReachDfs(self, arr: List[int], start: int) -> bool:
        if (start < 0 or start >= len(arr)) or arr[start] < 0:
            return False
        if arr[start] == 0:
            return True

        # Nodes are marked as visited, by setting their values to negative
        arr[start] = - arr[start]
        # Left or right branch will return the result
        left_check = self.canReach(arr, start + arr[start])
        right_check = self.canReach(arr, start - arr[start])
        return left_check or right_check
