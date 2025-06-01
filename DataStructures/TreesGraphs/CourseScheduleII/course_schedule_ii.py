'''
Leetcode 210: Course Schedule II

There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.
For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

Example 1:
Input: numCourses = 2, prerequisites = [[1,0]]
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].
'''

'''
Use DFS -> Topological sort. Sort by reverse order of finish times of nodes
- Time Complexity: O(V+E) where V represents the number of vertices and E represents the number of edges.
Essentially we iterate through each node and each vertex in the graph once and only once.
- Space Complexity: O(V+E).
  - We use the adjacency list to represent our graph initially. The space occupied is defined by the number of edges because for each node as the key, we have all its adjacent nodes in the form of a list as the value. Hence, O(E)
  - Additionally, we apply recursion in our algorithm, which in worst case will incur O(E) extra space in the function call stack.
  - To sum up, the overall space complexity is O(V+E).
'''
from typing import List
from collections import defaultdict

class Solution:
    WHITE = 1
    GRAY = 2
    BLACK = 3

    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        # Adjacency list
        adj_list = defaultdict(list)

        # Generate adjacency list: Given [a_i, b_i] -> adj_list[b_i].append(a_i)
        for dest, src in prerequisites:
            adj_list[src].append(dest)

        topsort_order = []
        is_possible = True

        # By default all vertices are white
        color = {course: Solution.WHITE for course in range(numCourses)}

        # DFS helper
        def dfs(node: int) -> None:
            nonlocal is_possible
            # If cycle found, return
            if not is_possible:
                return

            # Start the recursion
            color[node] = Solution.GRAY

            # Traverse neighbour vertices
            for neighbour in adj_list[node]:
                if color[neighbour] == Solution.WHITE:
                    dfs(neighbour)
                elif color[neighbour] == Solution.GRAY:
                    is_possible = False
            # Recursion ends, mark node as black, add to stack
            # Added to topsort_order in increasing order of finish times (will reverse at the end).
            color[node] = Solution.BLACK
            topsort_order.append(node)

        for course in range(numCourses):
            if color[course] == Solution.WHITE:
                dfs(course)

        return topsort_order[::-1] if is_possible else []