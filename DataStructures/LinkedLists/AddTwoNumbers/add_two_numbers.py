'''
Leetcode 2: Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.
'''
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode(0)
        curr = dummy_head
        carry = 0

        while l1 or l2 or carry != 0:
            # l1 / l2 values
            l1_val = l1.val if l1 else 0
            l2_val = l2.val if l2 else 0
            # sum column, calculate carry, create new node
            column_sum = l1_val + l2_val + carry
            carry = column_sum // 10
            new_node = ListNode(column_sum % 10)
            # Update node pointers
            curr.next = new_node
            curr = new_node
            # Move l1 / l2 pointers
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None

        return dummy_head.next