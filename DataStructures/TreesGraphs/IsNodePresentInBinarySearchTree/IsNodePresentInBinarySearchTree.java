/**
 * Check if a node exists in a binary search tree.
 */

import java.io.*;
import java.util.*;
import java.util.regex.*;
import java.text.*;
import java.math.*;


public class isNodePresentInBinarySearchTree {

    /**
     * Internal binary node class
     */
    private static class Node {
        Node left, right;
        int data;

        Node(int newData) {
            left = right = null;
            data = newData;
        }
    }

    /**
     * Insert a new node with <int></data> into binary search tree with root <Node><node/>
     * @param node
     * @param data
     * @return
     */
    private static Node insert(Node node, int data) {
        if (node==null) {
            node = new Node(data);
        }
        else {
            if (data <= node.data) {
                node.left = insert(node.left, data);
            }
            else {
                node.right = insert(node.right, data);
            }
        }
        return(node);
    }

    /**
     * Check if <int><val/> is present in binary search tree rooted at <Node><root/>
     * @param root
     * @param val
     * @return
     */
    private static boolean isPresent(Node root, int val){
        boolean result = false; // default is false.
        if(root == null){
            return result;
        } else {
            if(root.data == val) {
                result = true;
            } else if(root.data > val) {
                result = isPresent(root.left, val);
            } else if(root.data < val) {
                result = isPresent(root.right, val);
            }
        }
        return result;
    }

    /**
     * Create an example BST:
     *        10
     *       /  \
     *      5   15
     *     / \
     *    3  7
     *      / \
     *     6  8
     * @return root node of example BST
     */
    public static Node createExampleBst1(){
        Node root = new Node(10);
        Node rootLeft = new Node(5);
        Node rootRight = new Node(15);
        root.left = rootLeft;
        root.right = rootRight;

        Node rootLeftLeft = new Node(3);
        Node rootLeftRight = new Node(7);
        rootLeft.left = rootLeftLeft;
        rootLeft.right = rootLeftRight;

        rootLeftRight.left = new Node(6);
        Node rootLeftRightRight = new Node(8);
        rootLeftRight.right = rootLeftRightRight;

        return root;
    }

    public static void runTestCase(Node node, int k) {
        System.out.println("Is " + k + " present in binary search tree?: " + isPresent(node, k));
    }


    public static void main(String [] args) throws Exception{
        Node bst = createExampleBst1(); // [3,5,6,7,8,10,15]

        // Test case 1: k = 7, bst
        int k = 7;
        runTestCase(bst, k);

        // Test case 2: k = 4, bst
        k = 4;
        runTestCase(bst, k);
    }
}