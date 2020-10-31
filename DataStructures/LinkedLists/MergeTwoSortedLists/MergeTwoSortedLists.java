/**
 Merge two sorted linked lists and return it as a new list.
 The new list should be made by splicing together the nodes of the first two lists.
 */

/**
 * Links:
 * 1) http://techieme.in/merging-two-sorted-singly-linked-list/
 * 2) Solution using dummy nodes: http://www.geeksforgeeks.org/merge-two-sorted-linked-lists/ (method 1)
 */

/**
 * Definition for singly-linked list
 * Warning: This class is not public, because of "one public class per file" rule in Java
 */
class ListNode {
    int val;
    ListNode next;
    ListNode(int x){
        val = x;
    }
}

public class MergeTwoSortedLists {
    /**
     * Solution 1: Iterative solution, a lot of corner cases. My solution.
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode mergeTwoSortedLists1(ListNode l1, ListNode l2) {
        // Two pointers: One for the head of the result list, one for the runner.
        // WARNING: Assign these pointers to null. If you don't, this returns
        // "variable [head|runner] might not have been initialized" error.
        ListNode head = null;
        ListNode runner = null;

        /**
         * Find the head of the list
         */
        // If both lists are non-null, compare values, and assign the head
        if(l1 != null && l2 != null){
            // Assign the runner base
            head = ( (l1.val<l2.val) ? l1 : l2 );
            if(head == l1){
                l1 = l1.next;
            }
            if(head == l2){
                l2 = l2.next;
            }
        } // If only l1 is non-null, return l1
        else if(l1 != null){
            return l1;
        } // If only l2 is non-null, return l2
        else if(l2 != null){
            return l2;
        }

        /**
         * Assign runner to head
         */
        runner = head;

        /**
         * Iterate over lists, moving runner forward
         */
        // If both lists are non-null, compare values, assign runner
        while(l1 != null && l2 != null){
            runner.next = (l1.val < l2.val) ? l1 : l2;
            if(l1.val < l2.val){
                l1 = l1.next;
            } else if(l1.val >= l2.val){
                l2 = l2.next;
            }
            runner = runner.next;
        }
        // One of the lists reached the end at this point. Iterate over the other list
        // If l1 is not null, assign runner to the rest of list l1
        while(l1 != null){
            runner.next = l1;
            runner = runner.next;
            l1 = l1.next;
        }
        // If l2 is not null, assign runner to the rest of list l1
        while(l2 != null){
            runner.next = l2;
            runner = runner.next;
            l2 = l2.next;
        }

        // Return the head of the result list
        return head;
    }

    /**
     * Solution 2: Iterative solution, shortened by using a dummy node.
     * The idea here is to use a temporary dummy node as the start of the result list.
     * The use of dummy node removes the need for the first code block in Solution 1, which is to "Find the head of
     * the list". Then, the loop proceeds, removing one node from either ‘list1’ or ‘list2’, and adding it to tail. When
     * we are done, the result is in dummy.next.
     * Summary: Optimization here is to use a dummy variable to remove the code block for head assignment (handling nulls).
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode mergeTwoSortedLists2(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }

        // Two pointers: One for the dummy node, one for the runner.
        ListNode dummy = new ListNode(0);
        ListNode runner = dummy;

        /**
         * Iterate over lists, moving runner forward
         */
        // If both lists are non-null, compare values, assign runner
        while(l1 != null && l2 != null){
            runner.next = (l1.val < l2.val) ? l1 : l2;
            if(l1.val < l2.val){
                l1 = l1.next;
            } else if(l1.val >= l2.val){
                l2 = l2.next;
            }
            runner = runner.next;
        }
        // One of the lists reached the end at this point. Iterate over the other list
        // If l1 is not null, assign runner to the rest of list l1
        while(l1 != null){
            runner.next = l1;
            runner = runner.next;
            l1 = l1.next;
        }
        // If l2 is not null, assign runner to the rest of list l1
        while(l2 != null){
            runner.next = l2;
            runner = runner.next;
            l2 = l2.next;
        }

        // Return the head of the result list
        return dummy.next;
    }

    /**
     * Solution 3: Recursive solution, short.
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode mergeTwoSortedLists3(ListNode l1, ListNode l2) {
        // If any of the lists is null, return the other as the result
        if(l1 == null){
            return l2;
        }
        if(l2 == null){
            return l1;
        }

        // Based on values, assign the head, and assign the next node with the recursive call
        ListNode head = null;
        if(l1.val < l2.val){
            head = l1;
            head.next = mergeTwoSortedLists3(l1.next, l2);
        } else{
            head = l2;
            head.next = mergeTwoSortedLists3(l1, l2.next);
        }
        // Return head assigned in the first call to this function
        return head;
    }

    /**
     * Solution 4: Recursive solution, short, one less variable compared to solution 3 above.
     * @param l1
     * @param l2
     * @return
     */
    public static ListNode mergeTwoSortedLists4(ListNode l1, ListNode l2) {
        // If any of the lists is null, return the other as the result
        if(l1 == null) {
            return l2;
        }
        if(l2 == null) {
            return l1;
        }

        // Based on values, assign the next node with the recursive call first, then return the head of the corresponding list
        // Warning: We are not using a "head" variable, which is different from Solution 3 above (one less variable).
        if(l1.val < l2.val){
            l1.next = mergeTwoSortedLists4(l1.next, l2);
            return l1;
        }
        else{
            l2.next = mergeTwoSortedLists4(l1, l2.next);
            return l2;
        }
    }

    /**
     * Generate a list from an integer array
     * @param values
     * @return
     */
    public static ListNode generateList(int[] values){
        ListNode head = null;
        ListNode runner = null;

        // If array is empty, return null
        if(values.length == 0){
            return null;
        }
        // If array is non-empty, assign head to first node
        head = new ListNode(values[0]);
        // Assign runner to head, and iterate over the array, while filling the list with values
        runner = head;
        for(int i = 1; i < values.length; i++){
            ListNode newNode = new ListNode(values[i]);
            runner.next = newNode;
            runner = runner.next;
        }
        return head;
    }

    /**
     * Print the contents of a list given the head
     * @param head
     */
    public static String getListContent(ListNode node){
        String listContent = "[";
        while(node != null){
            listContent += (node.val + ", ");
            node = node.next;
        }
        listContent += "]";
        return listContent;
    }

    /**
     * Run a test case with all solutions
     * @param list1
     * @param list2
     */
    public static void runTestCase(int[] values1, int[] values2){
        System.out.println("mergeTwoSortedLists1: " + getListContent(mergeTwoSortedLists1(generateList(values1), generateList(values2))));
        System.out.println("mergeTwoSortedLists2: " + getListContent(mergeTwoSortedLists2(generateList(values1), generateList(values2))));
        System.out.println("mergeTwoSortedLists3: " + getListContent(mergeTwoSortedLists3(generateList(values1), generateList(values2))));
        System.out.println("mergeTwoSortedLists4: " + getListContent(mergeTwoSortedLists4(generateList(values1), generateList(values2))));
    }

    public static void main(String[] args) {
        // Test 1: Normal case
        int[] values1 = new int[]{1, 2, 3, 4, 5};
        int[] values2 = new int[]{-1, 80, 90, 100};
        runTestCase(values1, values2);

        // Test 2: Both empty lists
        int[] values3 = new int[]{};
        int[] values4 = new int[]{};
        runTestCase(values3, values4);

        // Test 3: Only the first list is empty
        int[] values5 = new int[]{};
        int[] values6 = new int[]{2, 3, 89};
        runTestCase(values5, values6);

        // Test 4: Only the second list is empty
        int[] values7 = new int[]{1, 4, 10};
        int[] values8 = new int[]{};
        runTestCase(values7, values8);

        // Test 5: Lists with elements that are equal
        int[] values9 = new int[]{1, 1, 2, 3, 4};
        int[] values10 = new int[]{0, 1, 2, 2, 20, 90};
        runTestCase(values9, values10);

        // Test 6: Lists with exact same elements in both
        int[] values11 = new int[]{1, 1, 1, 1};
        int[] values12 = new int[]{1, 1};
        runTestCase(values11, values12);

        // Test 7: Lists with exact same elements in both
        int[] values13 = new int[]{1, 1, 1, 1};
        int[] values14 = new int[]{2, 2};
        runTestCase(values13, values14);
    }
}