'''
Leetcode 21: Merge Two Sorted Lists

You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
'''

from typing import Optional

# Definition for singly-linked list.

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_node: Optional[ListNode] = ListNode(-1, None)
        merged_list = dummy_node
        # When both lists are non-empty, compare values and merge accordingly
        while list1 and list2:
            if list1.val <= list2.val:
                merged_list.next = list1
                list1 = list1.next
            else:
                merged_list.next = list2
                list2 = list2.next
            # Increment merged_list
            merged_list = merged_list.next
        # If elements in list1 exist (meaning none in list2), append all elements of list1
        if list1:
            while list1:
                merged_list.next = list1
                list1 = list1.next
                merged_list = merged_list.next
        # If elements in list2 exist (meaning none in list1), append all elements of list2
        if list2:
            while list2:
                merged_list.next = list2
                list2 = list2.next
                merged_list = merged_list.next

        # If merged_list is non-null, return dummy_node.next, else return empty list (None).
        return dummy_node.next if merged_list else None