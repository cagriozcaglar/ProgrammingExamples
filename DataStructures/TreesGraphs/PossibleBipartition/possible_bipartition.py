'''
Leetcode 886: Possible Bipartition

We want to split a group of n people (labeled from 1 to n) into two groups of any size.
Each person may dislike some other people, and they should not go into the same group.

Given the integer n and the array dislikes where dislikes[i] = [ai, bi] indicates that
the person labeled ai does not like the person labeled bi, return true if it is possible
to split everyone into two groups in this way.
'''

from collections import deque
from typing import List

class Solution:
    # BFS
    # Time Complexity: O(N+E)
    # Space Complexity: O(N+E)
    def possibleBipartitionBfs(self, n: int, dislikes: List[List[int]]) -> bool:
        # BFS. Returns false if there is color conflict.
        def bfs(source):
            q = deque([source])
            color[source] = 0  # Start with marking source as RED
            while q:
                node = q.popleft()
                for neighbour in adj[node]:
                    # If there is color conflict, return false
                    if color[neighbour] == color[node]:
                        return False
                    if color[neighbour] == -1:
                        color[neighbour] = 1 - color[node]  # Change color 0,1
                        q.append(neighbour)
            return True

        adj = [[] for _ in range(n+1)]
        for dislike in dislikes:
            adj[dislike[0]].append(dislike[1])
            adj[dislike[1]].append(dislike[0])

        # Color: 0 for red, 1 for blue, -1 for unassigned
        color = [-1] * (n+1)
        for i in range(1, n+1):  # 1 to n
            if color[i] == -1:
                # For each pending component, run BFS.
                if not bfs(i):
                    # Return false, if there is conflict in the component.
                    return False

        return True
        
    # DFS
    # Time Complexity: O(N+E)
    # Space Complexity: O(N+E)
    def possibleBipartitionDfs(self, n: int, dislikes: List[List[int]]) -> bool:
        def dfs(node, node_color):
            color[node] = node_color
            for neighbour in adj[node]:
                if color[neighbour] == color[node]:
                    return False
                if color[neighbour] == -1:
                    if not dfs(neighbour, 1- node_color):
                        return False
            return True

        adj = [[] for _ in range(n+1)]
        for dislike in dislikes:
            adj[dislike[0]].append(dislike[1])
            adj[dislike[1]].append(dislike[0])

        # Color: 0 for red, 1 for blue, -1 for unassigned
        color = [-1] * (n+1)
        for i in range(1, n+1):  # 1 to n
            if color[i] == -1:
                # For each pending component, run DFS, start with color 0.
                if not dfs(i, 0):
                    # Return false, if there is conflict in the component.
                    return False

        return True
