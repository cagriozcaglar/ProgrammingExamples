/**
 Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function
 to find the number of connected components in an undirected graph.

 Example 1:
 0         3
 |         |
 1 -- 2    4
 Given n = 5 and edges = [[0, 1], [1, 2], [3, 4]], return 2.
 */

import java.util.*;

public class NumberOfConnectedComponents {

    /**
     * Count the number of connnected components using DFS.
     * Uses:
     * 1) Stack<Integer> for DFS traversal,
     * 2) Vector<Boolean> to keep visited status of nodes,
     * 3) Vector<Vector<Integer>> to keep adjacency list of the graph
     */
    public static int numberOfConnectedComponents(int n, int[][] edges) {
        // Stack to implement DFS traversal
        Stack<Integer> nodeStack = new Stack<Integer>();
        // Boolean vector of visited indicator
        Vector<Boolean> visited = new Vector<Boolean>(n);
        // Graph adjacency list: from node to all neighbours
        Vector<ArrayList<Integer>> adjList = new Vector<ArrayList<Integer>>();
        for(int i = 0; i < n; i++){
            adjList.add(new ArrayList<Integer>()); // Initialize adjacency list
            visited.add(false);                    // Initialize visited Vector
        }
        // System.out.println("adjList.size(): " + adjList.size());
        // System.out.println("visited.size(): " + visited.size());

        int connectedComponentCount = 0;

        // Populate graph adjacency list using edges array
        for(int edgeIndex = 0; edgeIndex < edges.length; edgeIndex++) {
            int srcNodeIndex = edges[edgeIndex][0];
            int destNodeIndex = edges[edgeIndex][1];
            adjList.get(srcNodeIndex).add(destNodeIndex);
            adjList.get(destNodeIndex).add(srcNodeIndex);
        }

        // Traverse over the nodes using DFS to find connected components
        for(int i = 0; i < n; i++) {
            if(!visited.get(i)) {
                // If node is not visited, increment number of connected components, push node to stack
                connectedComponentCount++;
                nodeStack.push(i);

                while(!nodeStack.isEmpty()) {
                    // Remove top element from node stack, set it to visited
                    int current = nodeStack.peek();
                    nodeStack.pop();
                    visited.set(current, true);

                    // Traverse neighbours of the current node
                    for(int neighbourNodeIndex : adjList.get(current)) {
                        if(!visited.get(neighbourNodeIndex)) {
                            nodeStack.push(neighbourNodeIndex);
                        }
                    }
                }
            }
        }
        return connectedComponentCount;
    }

    /**
     * Run a test case with all solutions
     * @param n
     * @param edges
     */
    public static void runTestCases(int n, int[][] edges) {
        System.out.println("Number of connected components: " + numberOfConnectedComponents(n, edges));
    }

    public static void main(String[] args){
        // Test 1
        /**
        0           3
        |           |
        1 --- 2     4
        */
        int n = 5;
        int[][] edges = {{0, 1},{1, 2},{3,4}};
        runTestCases(n, edges);

        // Test 2: Case where a node is disconnected from all other nodes
        /**
         0    4--5    6
         |
         1    2--3
         */
        n = 7;
        int[][] edges2 = {{0, 1},{2, 3},{4,5}};
        runTestCases(n, edges2);
    }
}