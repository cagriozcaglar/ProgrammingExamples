"""
323. Number of Connected Components in an Undirected Graph
You have a graph of n nodes. You are given an integer n and an array edges where edges[i] = [ai, bi] indicates that
there is an edge between ai and bi in the graph.
Return the number of connected components in the graph.
"""
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        def dfs(node, adj, visited):
            if visited[node]:
                return
            visited[node] = 1
            for x in adj[node]:
                dfs(x, adj, visited)

        visited = [0] * n
        adj = defaultdict(list)
        for x,y in edges:
            adj[x].append(y)
            adj[y].append(x)

        count = 0
        for node in range(n):
            if not visited[node]:
                dfs(node, adj, visited)
                count += 1
        return count