'''
Leetcode 2672: Number of adjacent elements with same color

You are given an integer n representing an array colors of length n where all elements are set to 0's meaning uncolored. You are also given a 2D integer array queries where queries[i] = [indexi, colori]. For the ith query:
1) Set colors[indexi] to colori.
2) Count adjacent pairs in colors set to the same color (regardless of colori).

Return an array answer of the same length as queries where answer[i] is the answer to the ith query.
 
Example 1:
Input: n = 4, queries = [[0,2],[1,2],[3,1],[1,1],[2,1]]
Output: [0,1,1,0,2]
Explanation:
Initially array colors = [0,0,0,0], where 0 denotes uncolored elements of the array.
After the 1st query colors = [2,0,0,0]. The count of adjacent pairs with the same color is 0.
After the 2nd query colors = [2,2,0,0]. The count of adjacent pairs with the same color is 1.
After the 3rd query colors = [2,2,0,1]. The count of adjacent pairs with the same color is 1.
After the 4th query colors = [2,1,0,1]. The count of adjacent pairs with the same color is 0.
After the 5th query colors = [2,1,1,1]. The count of adjacent pairs with the same color is 2.
'''
from typing import List

class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        colors = [0] * n
        result = []
        adj_pair_count = 0

        for index, color in queries:
            # 1. Before color change
            # 1.1. Left: same color -> decrement
            if index-1 >= 0 and colors[index] != 0 and colors[index] == colors[index-1]:
                adj_pair_count -= 1
            # 1.2. Right: same color -> decrement
            if index+1 < n and colors[index] != 0 and colors[index] == colors[index+1]:
                adj_pair_count -= 1

            # Change the color
            colors[index] = color

            # 2. After color change
            # 2.1. Left: same color -> increment
            if index-1 >= 0 and colors[index] == colors[index-1]:
                adj_pair_count += 1
            # 2.2. Right: same color -> increment
            if index+1 < n and colors[index] == colors[index+1]:
                adj_pair_count += 1

            result.append(adj_pair_count)

        return result