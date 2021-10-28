"""
Given head which is a reference node to a singly-linked list. The value of each node in the linked list is either 0 or 1. The linked list holds the binary representation of a number.

Return the decimal value of the number in the linked list.

Note: This is a Leetcode question: https://leetcode.com/explore/challenge/card/november-leetcoding-challenge/564/week-1-november-1st-november-7th/3516/

Constraints:

The Linked List is not empty.
Number of nodes will not exceed 30.
Each node's value is either 0 or 1.

Hint 1: Traverse the linked list and store all values in a string or array. convert the values obtained to decimal value.

Hint 2: You can solve the problem in O(1) memory using bits operation. use shift left operation ( << ) and or operation
        ( | ) to get the decimal value in one operation.

Example 1:
Input: head = [1,0,1]
Output: 5
Explanation: (101) in base 2 = (5) in base 10

Example 2:
Input: head = [0]
Output: 0

Example 3:
Input: head = [1]
Output: 1

Example 4:
Input: head = [1,0,0,1,0,0,1,1,1,0,0,0,0,0,0]
Output: 18880

Example 5:
Input: head = [0,0]
Output: 0
"""

from typing import List

class ListNode:
    """
    Definition for singly-linked list.
    """
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def convertIntegerArrayToLinkedListAndReturnHead(elements: List[int]):
    if len(elements) == 0:
        return None
    head: ListNode = ListNode(elements[0])
    tempNode: ListNode = head
    for i in range(1, len(elements)):
        # Create new node using elements[i]
        newNode = ListNode(elements[i])
        # Set next pointer of tempNode to newNode
        tempNode.next = newNode
        # Before exiting the loop set tempNode to newNode for next iteration
        tempNode = newNode
    return head


class ConvertBinaryNumberInLinkedListToInteger:
    @staticmethod
    def getDecimalValue(head: ListNode) -> int:
        decimalValue = 0
        while head is not None: # until the end if linked list
            # Shift current value to left by 1 (multiply by 2), and add head.val to last bit (sum)
            decimalValue = (decimalValue << 1) | head.val
            # DO NOT FORGET TO MOVE THE POINTER TO NEXT ELEMENT IN THE LINKED LIST
            head = head.next
        return decimalValue


if __name__ == "__main__":
    """
    Example 1:
    Input: head = [1,0,1]
    Output: 5
    Explanation: (101) in base 2 = (5) in base 10
    """
    binaryValues: List[int] = [1, 0, 1]
    head: ListNode = convertIntegerArrayToLinkedListAndReturnHead(binaryValues)
    print(f"Decimal version of {binaryValues} is {ConvertBinaryNumberInLinkedListToInteger.getDecimalValue(head)}")

    """
    Example 2:
    Input: head = [0]
    Output: 0
    """
    binaryValues: List[int] = [0]
    head: ListNode = convertIntegerArrayToLinkedListAndReturnHead(binaryValues)
    print(f"Decimal version of {binaryValues} is {ConvertBinaryNumberInLinkedListToInteger.getDecimalValue(head)}")

    """
    Example 3:
    Input: head = [1]
    Output: 1
    """
    binaryValues: List[int] = [1]
    head: ListNode = convertIntegerArrayToLinkedListAndReturnHead(binaryValues)
    print(f"Decimal version of {binaryValues} is {ConvertBinaryNumberInLinkedListToInteger.getDecimalValue(head)}")

    """
    Example 4:
    Input: head = [1,0,0,1,0,0,1,1,1,0,0,0,0,0,0]
    Output: 18880
    """
    binaryValues: List[int] = [1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    head: ListNode = convertIntegerArrayToLinkedListAndReturnHead(binaryValues)
    print(f"Decimal version of {binaryValues} is {ConvertBinaryNumberInLinkedListToInteger.getDecimalValue(head)}")

    """
    Example 5:
    Input: head = [0,0]
    Output: 0
    """
    binaryValues: List[int] = [0,0]
    head: ListNode = convertIntegerArrayToLinkedListAndReturnHead(binaryValues)
    print(f"Decimal version of {binaryValues} is {ConvertBinaryNumberInLinkedListToInteger.getDecimalValue(head)}")
