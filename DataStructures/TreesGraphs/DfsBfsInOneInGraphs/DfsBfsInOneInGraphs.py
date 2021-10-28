from typing import List, Tuple, Dict

DEBUG_MODE = True


class Node:
    """
    Node class
    """
    def __init__(self, data: int):
        self.data = data
        self.children: List[int] = []

    def addChild(self, node: int):
        self.children.append(node)

    def getChildren(self):
        return self.children

    def __str__(self) -> str:
        return f"{self.data} -> ({self.children})"


class Edge:
    """
    Edge class
    """
    def __init__(self, src: Node, dest: Node):
        self.src = src
        self.dest = dest


class Graph:
    """
    Graph class
    """
    def __init__(self, nodes: List[int], edges: List[Tuple[int, int]] = None):
        self.nodes = nodes
        self.edges = edges
        self.valueToNodeMap: Dict[int, Node] = self.getValueToNodeMap()
        if DEBUG_MODE:
            for key, value in self.valueToNodeMap.items():
                print(f"key: {key}, value: {value}")
        self.addEdges(self.valueToNodeMap)
        if DEBUG_MODE:
            for key, value in self.valueToNodeMap.items():
                print(f"key: {key}, value: {value}")
        for nodeValue, nodeObject in self.getValueToNodeMap().items():
            print(nodeObject.__str__())

    def getValueToNodeMap(self) -> Dict[int, Node]:
        """
        Return the map from node values to nodes themselves. E.g. {1 -> Node(1),...}
        :return:
        """
        nodeObjectList: List[Node] = [Node(nodeValue) for nodeValue in self.nodes]
        return dict(zip(self.nodes, nodeObjectList))

    def addEdges(self, valueToNodeMap: Dict[int, Node]) -> None:
        if self.edges is not None:
            for edge in self.edges:
                src = edge[0]
                dest = edge[1]
                # print(f"src: {src}, dest: {dest}")
                # print("Before: " + valueToNodeMap[src].__str__())
                valueToNodeMap[src].addChild(dest)
                # print("After: " + valueToNodeMap[src].__str__())
                # for key, value in self.valueToNodeMap.items():
                #     print(f"key: {key}, value: {value}")
                # print("")
                # The next statement is only true if the graph is undirected. Omitting.
                # valueToNodeMap[dest].addChild(src)


class DfsBfsInOneInGraphs:
    @staticmethod
    def dfsRecursive(graph: Graph) -> List[int]:
        pass

    @staticmethod
    def dfsIterative(graph: Graph) -> List[int]:
        pass

    @staticmethod
    def bfs(graph: Graph) -> None:
        pass


if __name__ == "__main__":
    """
    Form a graph, and iterate over its values
    """
    nodes: List[int] = [0, 1, 2, 3, 4, 5]
    edges: List[Tuple[int, int]] = [(0, 1), (0, 4), (0, 5), (1, 3), (1, 4), (2, 1), (3, 2), (3, 4)]
    graph: Graph = Graph(nodes, edges)
    # nodes2: List[int] = [0, 1]
    # edges2: List[Tuple[int, int]] = [(0, 1)]
    # graph2: Graph = Graph(nodes2, edges2)
    for node in graph.nodes:
        print(node)
    for nodeValue, nodeObject in graph.valueToNodeMap.items():
        print(nodeObject.__str__())
