import java.util.*;

class depthFirstSearch
{
    // Number of vertices / nodes
    private int V;
    // Array of lists for Adjacency list representation
    private LinkedList<Integer>[] adj;

    // Constructor
    depthFirstSearch(int v)
    {
	V = v;
	adj = new LinkedList[v];
	// Initialize adjancey list for each node
	for(int i=0; i<v; i++)
	{
	    adj[i] = new LinkedList<Integer>();
	}
    }

    // Function to add an edge to the graph
    void addEdge(int node1, int node2)
    {
	// Add node2 to node1's list.
	adj[node1].add(node2);
    }

    // A function used by DFS
    void dfsUtil(int v, boolean[] visited)
    {
	// Visit the current node and print
	visited[v] = true;
	System.out.println(v + "");

	// Call dfsUtil for all neighbours of the current vertex
	Iterator<Integer> i = adj[v].listIterator();
	while(i.hasNext())
	{
	    int n = i.next();
	    if(!visited[n])
		dfsUtil(n, visited);
	}
	
    }

    // The function to do the DFS traversal, which uses recursive DFSUtil
    void dfs()
    {
	// Initialize all vertices as not visited
	// By default, Java initializes the boolean variables to false.
	boolean[] visited =  new boolean[V];

	// For each vertex, call recursive DFS-helper function to print DFS traversal
	for(int i=0; i<V; i++)
	    if(visited[i] == false)
		dfsUtil(i, visited);
    }

    // Example
    public static void main(String[] args)
    {
	depthFirstSearch g = new depthFirstSearch(7);
	g.addEdge(0, 1);
	g.addEdge(1, 2);
	g.addEdge(1, 3);
	g.addEdge(0, 4);
	g.addEdge(4, 5);
	g.addEdge(4, 6);

	g.dfs();
    }
    /* Input:
           0
	  / \
	 1   4
	/ \ / \
       2  3 5  6
     */
    /* Output:
       0
       1
       2
       3
       4
       5
       6
     */
}
