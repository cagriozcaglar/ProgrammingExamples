/**
 You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order
 and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

 You may assume the two numbers do not contain any leading zero, except the number 0 itself.

 Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
 Output: 7 -> 0 -> 8
 */


/**
 * Definition for singly-linked list.
 * Note: This class cannot be public, when there is another class that is declared as public
 * Therefore, "public" identifier is removed.
 */
class ListNode {
    int val;
    ListNode next;
    ListNode(int x) { val = x; }
}

/**
 * Main class for summing two lists
 */
public class SumLists {

    /**
     * Adds two numbers represented as lists in reverse order
     * It makes a call to a helper function which takes an additional parameter, carry, in order to use the carry value
     * in the sum of the next digit.
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        return addTwoNumbersHelper(l1, l2, 0);
    }

    /**
     * Adds two numbers represented as lists in reverse order.
     * This is a recursive function which takes two list nodes and a carry, adds list values if exists, omits the
     * addition if the node ends (the case when the lengths of numbers / lists are not equal).
     * Note 1: Carefully handle the condition where the length of numbers / lists are not equal. See how next node is
     * assigned below in the recursive call, making sure next node is passed as the argument if the node itself is not
     * null, and if the node is null, then null is passed as the argument.
     * @param l1
     * @param l2
     * @param carry
     * @return
     */
    public static ListNode addTwoNumbersHelper(ListNode l1, ListNode l2, int carry) {
        // If both lists are null and carry is 0, then return 0.
        if(l1 == null && l2 == null && carry == 0){
            return null;
        }

        // Initialize value
        int value = carry;

        // Add both list node values if exists
        if(l1 != null){
            value += l1.val;
        }
        if(l2 != null){
            value += l2.val;
        }

        // Calculate the remainder and carry
        int rem = value % 10;
        carry = value / 10;

        // Create the result head node of the list
        ListNode resultNode = new ListNode(rem);

        // Set the next node if exists, using recursion
        if(l1 != null || l2 != null){
            ListNode nextNode = addTwoNumbersHelper( (l1 == null ? null : l1.next), (l2 == null ? null : l2.next), carry);
            resultNode.next = nextNode;
        }

        // Return the head of the list
        return resultNode;
    }

    /**
     * Create a list of digits from array of digits. Used for testing.
     * @param int[] digits, an array of digits
     * @return ListNode, the head of list representation of digit array
     */
    public static ListNode createListFromDigitArray(int[] digits){
        if(digits.length == 0 || digits == null){
            return null;
        }
        ListNode result = new ListNode(digits[0]);
        ListNode head = result;
        ListNode nextNode;
        for(int i = 1; i < digits.length; i++){
            ListNode newNode = new ListNode(digits[i]);
            head.next = newNode;
            head = newNode;
        }
        return result;
    }

    /**
     * Given the head of digit list, print its contents
     * @param head
     */
    public static void printDigitListContents(ListNode head){
        while(head != null){
            System.out.print(head.val + " -> ");
            head = head.next;
        }
        System.out.println();
    }

    public static void main(String[] args){
        // Test 1: Lengths of numbers are equal
        // (2->4->3) + (5->6->4) = (7->0->8) ... => 342 + 465 = 807
        System.out.println("Test 1: Lengths of numbers are equal");
        int[] number1 = new int[]{2, 4, 3};
        int[] number2 = new int[]{5, 6, 4};
        // Create lists from digit arrays
        ListNode listNumber1 = createListFromDigitArray(number1);
        ListNode listNumber2 = createListFromDigitArray(number2);
        // Print contents of both lists
        printDigitListContents(listNumber1);
        printDigitListContents(listNumber2);
        // Add the lists
        ListNode listSum = addTwoNumbers(listNumber1, listNumber2);
        printDigitListContents(listSum);

        // Test 2: Lengths of numbers are unequal
        // (1->2->4->3) + (5->6->4) = (6->8->8->3) ... => 3421 + 465 = 3886
        System.out.println("Test 2: Lengths of numbers are unequal");
        int[] number3 = new int[]{1, 2, 4, 3};
        int[] number4 = new int[]{5, 6, 4};
        // Create lists from digit arrays
        ListNode listNumber3 = createListFromDigitArray(number3);
        ListNode listNumber4 = createListFromDigitArray(number4);
        // Print contents of both lists
        printDigitListContents(listNumber3);
        printDigitListContents(listNumber4);
        // Add the lists
        ListNode listSum2 = addTwoNumbers(listNumber3, listNumber4);
        printDigitListContents(listSum2);
    }
}