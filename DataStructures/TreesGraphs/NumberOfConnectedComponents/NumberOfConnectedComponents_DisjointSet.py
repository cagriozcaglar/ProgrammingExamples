"""
Given n nodes labeled from 0 to (n-1), and a list of undirected edges where each edge is a pair of nodes,
write a function to find the number of connected components in an undirected graph

Example:
Given n = 5 and edges = [ [0,1], [1,2], [3,4] ]. the number of connected components is 2.

    0           3
    |           |
    1 --- 2     4
"""

"""
Solution with disjoint set
"""

class connectedComponents:
    """
    Constructor for connectedComponents class
    @param: n - number of nodes in the graph
    It also initializes the parents to [0, .., (n-1)] and number of connected components to n
    """
    def __init__(self, n):
        # Rank of all nodes: How many steps does it talke to get to the parent node in the disjoint set
        # Initialized to zero for all nodes
        self.rank = [0] * n
        # Parent of nodes: The list of parent nodes for each node in the disjoint set: 0 to (n-1)
        self.parent = range(n)
        # Number of connected components in the disjoint set
        self.numberOfConnectedComponents = n

    """
    Get the parent of a node: Follow links to find the parent    
    """
    def getParent(self, v):
        while self.parent[v] != v:
            v = self.parent[v];
            #self.parent[v] = self.getParent(v)
            #self.parent[v] = self.getParent(self.parent[v])
        return self.parent[v]

    """
    Count the number of components
    """
    def countComponents(self, edges):
        while edges and self.numberOfConnectedComponents > 1:
            edge = edges.pop() # Pop from last element in the list of edges
            v1 = edge[0]
            v2 = edge[1]
            p1 = self.getParent(v1)
            p2 = self.getParent(v2)
            # If nodes have the same parent and already part of same connected component, continue
            if p1 == p2:
                continue
            # If nodes do not have the same parent, meaning they are not in the same connected component
            else:
                if self.rank[p1] > self.rank[p2]:
                    self.parent[p2] = p1
                elif self.rank[p2] > self.rank[p1]:
                    self.parent[p1] = p2
                else:
                    self.parent[p1] = p2
                    self.rank[p1] += 1
                # We combined the connected components of two nodes. Therefore, decrease the number of connected components.
                self.numberOfConnectedComponents -= 1
        return self.numberOfConnectedComponents

if __name__ == "__main__":
    # Test 1
    """
    0           3
    |           |
    1 --- 2     4
    """
    n = 5
    edges = [ [0,1], [1,2], [3,4] ]
    connectedComponentsInstance = connectedComponents(n)
    print(connectedComponentsInstance.countComponents(edges))
    # Test 2
    """
    0     3 
    |     | 
    1 --- 2     4    5   6   7
    """
    n = 8
    edges = [ [0,1], [1,2], [2,3] ]
    connectedComponentsInstance = connectedComponents(n)
    print(connectedComponentsInstance.countComponents(edges))
    # Test 3
    """
    0   3 
    |   | 
    1 - 2
    """
    n = 4
    edges = [ [0,1], [1,2], [2,3] ]
    connectedComponentsInstance = connectedComponents(n)
    print(connectedComponentsInstance.countComponents(edges))
    # Test 4
    """
    0 - 3 
    | /   
    1 - 2
    """
    n = 4
    edges = [ [0,1], [1,2], [2,3], [1,3], [0,3]]
    connectedComponentsInstance = connectedComponents(n)
    print(connectedComponentsInstance.countComponents(edges))
