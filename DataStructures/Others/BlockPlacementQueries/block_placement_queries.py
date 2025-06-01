'''
Leetcode 3161: Block Placement Queries
There exists an infinite number line, with its origin at 0 and extending towards the positive x-axis.

You are given a 2D array queries, which contains two types of queries:

For a query of type 1, queries[i] = [1, x]. Build an obstacle at distance x from the origin. It is guaranteed that there is no obstacle at distance x when the query is asked.
For a query of type 2, queries[i] = [2, x, sz]. Check if it is possible to place a block of size sz anywhere in the range [0, x] on the line, such that the block entirely lies in the range [0, x]. A block cannot be placed if it intersects with any obstacle, but it may touch it. Note that you do not actually place the block. Queries are separate.
Return a boolean array results, where results[i] is true if you can place the block specified in the ith query of type 2, and false otherwise.
'''

'''
Idea: Use a union–find data structure (also known as DSU) to track the next available free index.
How It Works:
Initialization:
No positions are occupied initially.
Find Function:
Implement a find(x) function that returns the smallest available position at or after x. If x is free, then find(x) = x; if not, recursively find the next free position.
Placement (Union):
When a block is placed at position x, mark it as occupied by “unioning” it with x + 1. This means that for future queries, the next available position will be determined by find(x + 1).
Processing a Query:
For a query with desired index x, compute pos = find(x). Then place the block at pos and update the union–find structure by setting parent[pos] = pos + 1. Return pos as the answer for that query.
'''
def blockPlacementQueries(queries):
    # Dictionary to store the union-find parent pointer.
    parent = {}
    
    def find(x):
        # If x is not in parent, it's free and x is its own parent.
        if x not in parent:
            return x
        # Path compression: update parent[x] to the next free position.
        parent[x] = find(parent[x])
        return parent[x]
    
    def placeBlock(x):
        # Find the next available free position at or after x.
        pos = find(x)
        # Mark pos as occupied by linking it to pos+1.
        parent[pos] = pos + 1
        return pos
    
    # Process each query and record the resulting placement.
    result = []
    for q in queries:
        result.append(placeBlock(q))
    return result

# Main method to run and test the code.
if __name__ == "__main__":
    test_cases = [
        ([1, 2, 1, 2], [1, 2, 3, 4]),
        ([5, 5, 5], [5, 6, 7]),
        ([10], [10])
    ]
    for queries, expected in test_cases:
        res = blockPlacementQueries(queries)
        print(f"Queries: {queries} -> Placements: {res} (Expected: {expected})")