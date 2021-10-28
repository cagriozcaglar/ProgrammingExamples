"""
Route Between Nodes: Given a directed graph, design an algorithm to find out whether there is a route between two nodes.

Note: This is Question 4.1 in CTCI 6th edition.

Example solution: https://www.geeksforgeeks.org/find-if-there-is-a-path-between-two-vertices-in-a-given-graph/
"""
# NOTE: The "." in front of python file "Node"(.py) tells the compiler that Node.py is in the current directory.
#  Without this, you will get an error.
# from .Node import Node, State
from Node import Node, State
from collections import deque
from typing import List, Dict, Tuple


class Graph:
    """
    Graph class
    """
    def __init__(self):
        self.nodes: List[Node] = []

    def addNode(self, node: Node) -> None:
        self.nodes.append(node)

    def getNodes(self) -> List[Node]:
        return self.nodes

    def __str__(self) -> str:
        graphInfo = "Graph info: \n\t"
        nodeInfo: str = '\n\t'.join(node.__str__() for node in self.getNodes())
        return graphInfo + nodeInfo

    def routeBetweenNodes(self, source: Node, destination: Node) -> bool:
        """
        Given source and destination node in graph, return whether if there is a route from source to destination
        :param source:
        :param destination:
        :return:
        """
        if source == destination:
            return True
        # Mark all vertices of graph as not visited
        for node in self.getNodes():
            node.state = State.UNVISITED
        # Create BFS queue
        # NOTE: When initializing Python deque, note the empty list inside. Initialize using deque([]), not deque()
        bfsQueue: deque[Node] = deque([])
        # Push source node in BFS queue, set it to visited (append pushes to end)
        # ASSUMPTION: We append elements from the right, delete / pop elements from left (FIFO: First In, First Out)
        bfsQueue.append(source)
        source.state = State.VISITED

        # Iterate over the nodes in queue, until queue is empty and hence all nodes are visited (until we hit dest)
        # NOTE: There is no .isEmpty() or .empty() method for dequeues in Python, use len(bfsQueue) != 0 instead
        while len(bfsQueue) != 0:
            # Pop first element in queue
            currentNode = bfsQueue.popleft()
            # Go through neighbours of currentNode
            if currentNode:
                for neighborNode in currentNode.getChildren():
                    # If neighborNode is unvisited,
                    if neighborNode.state == State.UNVISITED:
                        # Check if is equal to dest
                        if neighborNode == destination:
                            return True
                        # Otherwise, push it to queue
                        else:
                            bfsQueue.append(neighborNode)
                            # Regardless or both cases, mark neighborNode as visited
                            neighborNode.state = State.VISITED
        return False


def createGraphFromNodesAndEdges(nodes: List[int], edges: Dict[int, List[int]]) -> Graph:
    """
    Helper function to create graphs
    Example use:
        nodes: List[int] = [0, 1, 2, 3, 4, 5]
        edges: Dict[int, List[int]] = {0: [1, 4, 5], 1: [3, 4], 2: [1], 3: [2, 4]}
        graph: Graph = createGraphFromNodesAndEdges(nodes, edges)
    """
    # graph2: Graph = Graph()
    # nodes: List[int] = [0, 1, 2, 3, 4, 5]
    # edges: Dict[int, List[int]] = {0: [1, 4, 5], 1: [3, 4], 2: [1], 3: [2, 4]}
    graph: Graph = Graph()
    # Create node objects, and a map from node values to node objects
    nodeObjects: List[Node] = [Node(data) for data in nodes]
    nodeValueToObjectMap = dict(zip(nodes, nodeObjects))
    # Add edges to nodes
    for nodeValue, nodeObject in nodeValueToObjectMap.items():
        neighbors: List[int] = edges.get(nodeValue)
        if neighbors:
            for neighborNodeValue in neighbors:
                neighborNode: Node = nodeValueToObjectMap[neighborNodeValue]
                nodeObject.addChild(neighborNode)
    # Add the nodes to the graph
    for nodeObject in nodeObjects:
        graph.addNode(nodeObject)
    # Return graph
    return graph


if __name__ == "__main__":
    graph: Graph = Graph()
    # Node 1
    node1: Node = Node(1)
    node1.addChild(Node(3))
    print(node1)  # Node: value = 1, state = State.UNVISITED, children = [3]
    # Node 2
    node2: Node = Node(2)
    node2.addChild(Node(4))
    print(node2)  # Node: value = 2, state = State.UNVISITED, children = [4]
    # Nodes
    nodes: List[Node] = [node1, node2]
    for node in nodes:
        graph.addNode(node)
    # Print graph info
    print(graph.__str__())
    '''
    Graph info: 
        Node: value = 1, state = State.UNVISITED, children = [3]
        Node: value = 2, state = State.UNVISITED, children = [4]
    '''

    """
    Graph Diagram:
    
    0 -> 1 <- 2
    | \  | \  ^
    v   vv   v|
    5    4 <- 3
    """
    # Basic graph info
    nodes: List[int] = [0, 1, 2, 3, 4, 5]
    edges: Dict[int, List[int]] = {0: [1, 4, 5], 1: [3, 4], 2: [1], 3: [2, 4]}
    graph2: Graph = createGraphFromNodesAndEdges(nodes, edges)

    # Print graph info
    print(graph2.__str__())
    '''
    Graph info: 
        Node: value = 0, state = State.UNVISITED, children = [1,4,5]
        Node: value = 1, state = State.UNVISITED, children = [3,4]
        Node: value = 2, state = State.UNVISITED, children = [1]
        Node: value = 3, state = State.UNVISITED, children = [2,4]
        Node: value = 4, state = State.UNVISITED, children = []
        Node: value = 5, state = State.UNVISITED, children = []
    '''

    # Test RouteBetweenNodes method
    sourceDestinationNodeIndexPairs: List[Tuple[int, int]] = [(0, 1), (2, 5), (4, 3)]
    for sourceDestinationIndexPair in sourceDestinationNodeIndexPairs:
        source: Node = graph2.getNodes()[sourceDestinationIndexPair[0]]
        destination: Node = graph2.getNodes()[sourceDestinationIndexPair[1]]
        print(f"There is a route between Node {source.data} and Node {destination.data}: "
              f"{graph2.routeBetweenNodes(source, destination)}")
    """
    There is a route between Node 0 and Node 1: True
    There is a route between Node 2 and Node 5: False
    There is a route between Node 4 and Node 3: False
    """
