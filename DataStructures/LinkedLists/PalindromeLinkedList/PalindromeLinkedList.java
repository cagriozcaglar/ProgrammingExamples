/**
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
 */

// Definition for singly-linked list.
public class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

class Solution {

    public boolean isPalindrome(ListNode head) {
        // Error check (TODO)
        if(head == null || head.next == null) {
            return true;
        }

        // Initialize slow and fast pointers
        ListNode slow = head;
        ListNode fast = head;

        // Initialize stack
        Stack<Integer> valueStack = new Stack<Integer>();

        while(fast != null && fast.next != null) {
            // Add element in slow pointer to stack
            valueStack.push(slow.val);

            // Move slow pointer by 1, fast pointer by 2
            slow = slow.next;
            fast = fast.next.next;
        }

        // Check if there are odd number of nodes
        // If fast is not null, pointers are at the following locations in a list of length 2n+1:
        // slow: n+1, fast: 2n+1, but stack has the first n elements (not (n+1)).
        // So, if number of nodes is odd, skip the middle element, by incrementing the slow pointer
        // Otherwise, there are even number of nodes, and there is no need to increment slow pointer
        if(fast != null) {
            slow = slow.next;
        }

        // Check equality between stack values (front of the list), and rest of the list using slow
        while(slow != null) {
            // If the value of slow pointer and value at the top of stack do not match, not a palindrome
            if(slow.val != valueStack.pop()) {
                return false;
            }
            slow = slow.next;
        }

        // If all false checks pass, return true
        return true;
    }
}