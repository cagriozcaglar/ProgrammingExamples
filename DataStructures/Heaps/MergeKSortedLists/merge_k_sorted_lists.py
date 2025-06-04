'''
Leetcode 23: Merge k Sorted Lists

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.
'''
from typing import List, Optional
import heapq

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Definition for singly-linked list.
class HeapNode:
    def __init__(self, node):
        self.node = node

    def __lt__(self, other):
        # Define comparison based on ListNode's value
        return self.node.val < other.node.val


class Solution:
    def mergeKLists(
        self, lists: List[Optional[ListNode]]
    ) -> Optional[ListNode]:
        dummy = ListNode(0)
        current = dummy
        heap = []

        # Initialize the heap
        for l in lists:
            if l:
                heapq.heappush(heap, HeapNode(l))

        # Extract the minimum node and add its next node to the heap
        while heap:
            heap_node = heapq.heappop(heap)
            node = heap_node.node
            current.next = node
            current = current.next
            if node.next:
                heapq.heappush(heap, HeapNode(node.next))

        return dummy.next