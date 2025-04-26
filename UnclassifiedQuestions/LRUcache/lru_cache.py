'''
Leetcode 146: LRU Cache

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

- LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
- int get(int key) Return the value of the key if the key exists, otherwise return -1.
- void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.

The functions get and put must each run in O(1) average time complexity.
'''
from collections import defaultdict, OrderedDict
from typing import Dict

# Solution 1: Implement yourself, using HashMap + Doubly Linked List
class ListNode:
    def __init__(self, key, val):
        # Doubly linked list node
        self.key = key
        self.val = val
        self.next = None
        self.prev = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic: Dict[int, ListNode] = defaultdict(ListNode)
        self.head = ListNode(-1, -1)
        self.tail = ListNode(-1, -1)
        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        node = self.dic[key]
        # Move the node to tail (first remove, then add).
        self.remove(node)
        self.add(node)
        return node.val


    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            old_node = self.dic[key]
            self.remove(old_node)
        
        node = ListNode(key, value)
        self.dic[key] = node
        self.add(node)

        if len(self.dic) > self.capacity:
            # Delete real head
            real_head = self.head.next
            self.remove(real_head)
            del self.dic[real_head.key]


    def add(self, node) -> None:
        real_tail = self.tail.prev
        # Update next / prev pointers of real_tail
        real_tail.next = node
        node.prev = real_tail
        # Update next / prev pointers of node
        node.next = self.tail
        self.tail.prev = node


    def remove(self, node) -> None:
        # Update next / prev pointers of node.prev and node.next respectively
        node.prev.next = node.next
        node.next.prev = node.prev


# Solution 2: Using built-in OrderedDict
class LRUCache2:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.dic = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.dic:
            return -1
        self.dic.move_to_end(key)
        return self.dic[key]
    
    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic.move_to_end(key)
        self.dic[key] = value
        if len(self.dic) > self.capacity:
            self.dic.popitem(False)

# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)