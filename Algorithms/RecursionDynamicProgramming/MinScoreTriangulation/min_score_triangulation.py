'''
From:
1) https://github.com/doocs/leetcode/blob/main/solution/1000-1099/1039.Minimum%20Score%20Triangulation%20of%20Polygon/README_EN.md
2) https://algo.monster/liteproblems/1039
3) https://leetcode.ca/2018-10-04-1039-Minimum-Score-Triangulation-of-Polygon/
'''
'''
We design a function $\text{dfs}(i, j)$, which represents the minimum score after triangulating the polygon from vertex $i$ to $j$. The answer is $\text{dfs}(0, n - 1)$.

The calculation process of $\text{dfs}(i, j)$ is as follows:

-   If $i + 1 = j$, it means the polygon has only two vertices and cannot be triangulated, so we return $0$;
-   Otherwise, we enumerate a vertex $k$ between $i$ and $j$, i.e., $i \lt k \lt j$. Triangulating the polygon from vertex $i$ to $j$ can be divided into two subproblems: triangulating the polygon from vertex $i$ to $k$ and triangulating the polygon from vertex $k$ to $j$. The minimum scores of these two subproblems are $\text{dfs}(i, k)$ and $\text{dfs}(k, j)$, respectively. The score of the triangle formed by vertices $i$, $j$, and $k$ is $\text{values}[i] \times \text{values}[k] \times \text{values}[j]$. Thus, the minimum score for this triangulation is $\text{dfs}(i, k) + \text{dfs}(k, j) + \text{values}[i] \times \text{values}[k] \times \text{values}[j]$. We take the minimum value of all possibilities, which is the value of $\text{dfs}(i, j)$.

To avoid repeated calculations, we can use memoization, i.e., use a hash table or an array to store the already computed function values.

Finally, we return $\text{dfs}(0, n - 1)$.

The time complexity is $O(n^3)$, and the space complexity is $O(n^2)$, where $n$ is the number of vertices in the polygon.
'''

from functools import cache
from typing import List

class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        @cache
        def dfs(i: int, j: int) -> int:
            if i+1 == j:
                return 0
            return min(
                dfs(i,k) + dfs(k,j) + values[i] * values[j] * values[k]
                for k in range(i+1,j)
            )
        
        return dfs(0, len(values)-1)