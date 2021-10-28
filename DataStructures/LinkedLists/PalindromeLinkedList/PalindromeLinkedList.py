"""
* Given a singly linked list, determine if it is a palindrome.
*
* Note: This is a Leetcode question: https://leetcode.com/problems/palindrome-linked-list/
*
* Example 1:
* Input: 1->2
* Output: false
*
* Example 2:
* Input: 1->2->2->1
* Output: true
*
* Follow up:
* Could you do it in O(n) time and O(1) space?
*
* Notes:
*  - Example solution: https://github.com/careercup/ctci/blob/master/java/Chapter%202/Question2_7/QuestionB.java
*  - Example solution: https://www.programcreek.com/2014/07/leetcode-palindrome-linked-list-java/
*  - Example solution: https://leetcode.com/problems/palindrome-linked-list/discuss/131028/JAVA-code-with-stack-in-O(n)-time
"""

from collections import deque
from typing import List, Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class PalindromeLinkedList:
    # Iterative Solution with Stack.
    # Uses slow and fast pointers to find the mid-point
    # Time complexity: O(n), Space complexity: O(n)
    @staticmethod
    def isPalindrome_IterativeWithStack(head: ListNode) -> bool:
        # Corner cases: If LL length is 0 or 1, it is a palindrome, return true
        if not head or not head.next:
            return True

        # Set slow (speed: v) and fast (speed: 2v) pointers to head
        slow: ListNode = head
        fast: ListNode = head

        # Initialize the stack to reverse the elements of first half of LL
        # Note: You can use the following in Python for stack data structure
        # 1) a list, or
        # 2) collections.deque
        # Both use the same append() and pop() methods
        stack = deque()

        # Proceed both pointers at their speed while both fast and fast.next are not null
        # Append data from slow pointer to stack, which will have reverse of first half
        while fast and fast.next:
            stack.append(slow.val)
            slow = slow.next
            fast = fast.next.next

        # If the list has odd number of elements, then:
        # fast != None && fast.next == None
        # If so, skip middle element by incrementing slow pointer.
        if fast:
            slow = slow.next

        # Compare equality between:
        # 1) Reverse of first half of LL: Stack content
        # 2) Second half of LL, starting with current location of slow pointer
        while slow:
            top: int = stack.pop()
            if slow.val != top:
                return False
            # Do not forget to increment the pointer (Add it when you open while loop)
            slow = slow.next

        # If no return false so far, all equality checks are correct, return true
        return True


class PalindromeLinkedList2:
    # Iterative Solution using in-place Linked list reversal. Space efficient.
    # Uses slow and fast pointers to find the mid-point. Reverses the second half, compares with first half.
    # Time complexity: O(n), Space complexity: O(n)
    @staticmethod
    def isPalindrome_IterativeSpaceEfficient(head: ListNode) -> bool:
        # Corner cases: If LL length is 0 or 1, it is a palindrome, return true
        if not head or not head.next:
            return True

        # Set slow (speed: v) and fast (speed: 2v) pointers to head
        slow: ListNode = head
        fast: ListNode = head

        # Proceed both pointers at their speed while both fast and fast.next are not null
        # Append data from slow pointer to stack, which will have reverse of first half
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        # If the list has odd number of elements, then:
        # fast != None && fast.next == None
        # If so, skip middle element by incrementing slow pointer.
        if fast:
            slow = slow.next

        # Compare equality between:
        # 1) Reverse of first half of LL: Stack content
        # 2) Second half of LL, starting with current location of slow pointer
        while slow:
            top: int = stack.pop()
            if slow.val != top:
                return False
            # Do not forget to increment the pointer (Add it when you open while loop)
            slow = slow.next

        # If no return false so far, all equality checks are correct, return true
        return True


def convertLinkedListToList(node: ListNode) -> List:
    """
    Given a linked list with its head "node", return the elements in the linked list as a list
    :param node:
    :return:
    """
    newList: List = []
    while node:
        newList.append(node.val)
        node = node.next
    return newList


def formLinkedListFromList(elements: List[int]) -> Optional[ListNode]:
    if len(elements) == 0:
        return None
    head: ListNode = ListNode(elements[0])
    runner: ListNode = head
    # elements=[1,2,3,4]
    # a = [element for index, element in enumerate(elements,4)]
    for index, element in enumerate(elements[1:]):
        newNode: ListNode = ListNode(element)
        runner.next = newNode
        # Do not forget to increment the pointer. Add this when you open the for loop
        runner = runner.next
    return head


if __name__ == "__main__":
    """
    * Example 1:
    * Input: 1->2
    * Output: false
    """
    list1 = [1, 2]
    linkedList1: ListNode = formLinkedListFromList(list1)
    print(f"Is {convertLinkedListToList(linkedList1)} a palindrome?: "
          f"{PalindromeLinkedList.isPalindrome_IterativeWithStack(linkedList1)}")

    """
    * Example 2:
    * Input: 1->2->2->1
    * Output: true
    """
    list2 = [1, 2, 2, 1]
    linkedList2: ListNode = formLinkedListFromList(list2)
    print(f"Is {convertLinkedListToList(linkedList2)} a palindrome?: "
          f"{PalindromeLinkedList.isPalindrome_IterativeWithStack(linkedList2)}")

    """
    * Example 3:
    * Input: 1->2->3->2->1
    * Output: true
    """
    list3 = [1, 2, 3, 2, 1]
    linkedList3: ListNode = formLinkedListFromList(list3)
    print(f"Is {convertLinkedListToList(linkedList3)} a palindrome?: "
          f"{PalindromeLinkedList.isPalindrome_IterativeWithStack(linkedList3)}")
