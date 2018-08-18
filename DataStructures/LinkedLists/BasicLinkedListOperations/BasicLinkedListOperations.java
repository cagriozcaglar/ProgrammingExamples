/**
 A Node class is provided for you in the editor. A Node object has an integer data field, "data", and a Node instance
 pointer, "next", pointing to another node (i.e.: the next node in a list).

 A Node insert function is also declared in your editor. It has two parameters: a pointer, "head", pointing to the first
 node of a linked list, and an integer "data" value that must be added to the end of the list as a new Node object.

 Task
 Complete the insert function in your editor so that it creates a new Node (pass "data" as the Node constructor argument)
 and inserts it at the tail of the linked list referenced by the "head" parameter. Once the new node is added, return
 the reference to the "head" node.
 Note: If the  argument passed to the insert function is null, then the initial list is empty.
 */

import java.io.*;
import java.util.*;

/**
 * Node class
 */
class Node {
    int data;
    Node next;

    Node(int d) {
        data = d;
        next = null;
    }
}

public class BasicLinkedListOperations {

    /**
     * Insert element at the end of the linked list
     */
    public static Node insert(Node head, int data) {
        Node newNode = new Node(data);
        // If the list is empty, and make the new node the head of the linked list
        if(head == null){
            return newNode;
        }
        else { // If the list is non-empty (head not null), iterate over the list until tail, insert the new node
               // to the end of the list, and return original head
            Node actualHead = head;
            while(head.next != null){
                head = head.next;
            }
            head.next = newNode;
            return actualHead;
        }
    }

    /**
     * Print contents of the linked list
     */
    public static void printContents(Node head) {
        Node start = head;
        while(start != null) {
            System.out.print(start.data + " ");
            start = start.next;
        }
        System.out.println();
    }

    public static void main(String args[]) {
        Node head = null;
        int[] array = {2, 3, 4, 1};
        for(int value: array){
            head = insert(head, value);
        }
        printContents(head);
    }
}