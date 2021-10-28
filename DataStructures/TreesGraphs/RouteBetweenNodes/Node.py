from __future__ import annotations
from enum import Enum
from typing import List


class State(Enum):
    UNVISITED = 0
    VISITING = 1
    VISITED = 2


class Node:
    # def __init__(self, data: int, state: State, children: List[Node] = []):
    def __init__(self, data: int, state: State, children: List[Node] = []):
        self.data = data
        self.state = state
        self.children: List[Node] = children

    def addChild(self, node: Node) -> None:
        self.children.append(node)

    def getChildren(self) -> List[Node]:
        return self.children

    def __str__(self):
        childrenSerialized = f"[{','.join([str(childNode.data) for childNode in self.getChildren()])}]"
        return f"Node: value = {self.data}, state = {self.state}, children = {childrenSerialized}"


if __name__ == "__main__":
    node1: Node = Node(1, State.UNVISITED)
    print(node1)
    # Node: value = 1, state = State.UNVISITED, children = []
    node1.addChild(Node(2, State.VISITING))
    print(node1)
    # Node: value = 1, state = State.UNVISITED, children = [2]
    node1.addChild(Node(3, State.UNVISITED))
    print(node1)
