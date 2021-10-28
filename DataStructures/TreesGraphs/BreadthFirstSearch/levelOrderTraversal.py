"""
Say we have a binary tree, write a function which does a level order traversal on the tree 
and prints a special character, '#' at the end of every level
"""

from collections import deque

class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.children = None
        self.data = data

def level_order_traversal(root):
    "empty queue to store BFS results"
    print("Here1")
    queue = deque()

    print("Here2")
    "initialize with root node"
    #queue.add(root)
    queue.appendleft(root)

    print("Here3")
    "buffer queue"
    buffer_queue = deque()

    print("Here4")
    "print root"
    print(root.data)
    print("#")

    print("Here5")
    "every node in the main queue"
    while(not queue and not buffer_queue):
        while(not queue):
            
            print("Here5.1")
            "deque-ing TreeNode"
            node = queue.popleft()
            
            print("Here5.2")
            "TreeNode.children() return list of all child TreeNodes"
            if(node.left):
                buffer_queue.appendleft(node.left)
            if(node.right):
                buffer_queue.appendleft(node.right)

        print("Here5.3")
        "print nodes in that level to console"
        print(''.join(list(buffer_queue)))
        print('#')

        print("Here5.4")
        "interchange buffer and main queue"
        buffer_queue, queue = queue, buffer_queue

"""
    5
   / \
  4   3
 / \ /
2  1 0
"""
if __name__ == '__main__':
    print "Nothing"
    root = Node(5)
    rootLeft = Node(4)
    rootRight = Node(3)
    root.left = rootLeft
    root.right = rootRight
    rootLeft.left = Node(2)
    rootLeft.right = Node(1)
    rootRight.left = Node(0)
    level_order_traversal(root)
    #print(root.left.data)
    #print(root.right.data)
