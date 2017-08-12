/**
 The height of a binary tree is the number of edges between the tree's root and its furthest leaf. This means that a tree containing a single node has a height of 0.

 Complete the getHeight function provided in your editor so that it returns the height of a binary tree. This function has a parameter, root , which is a pointer to the root node of a binary tree.

 Note: The Height of binary tree with single node is taken as zero.
 Note: A binary search tree is a binary tree in which the value of each parent node's left child is less than the value the parent node, and the value of the parent node is less than the value of its right child.
**/

import java.util.*;
import java.io.*;

class Node {
    Node left;
    Node right;
    int data;

    Node(int data) {
        this.data = data;
        left = null;
        right = null;
    }
}

class heighOfBinaryTree {
    static int height(Node root) {
        if (root == null) {
            // System.out.println("Null checker: root null");
            // Return -1 instead of 0: Becaue the height of a binary tree with only root node is 0.
            return -1;
        } else {
            // System.out.println("Else: " + root.data);
            return (1 + Math.max(height(root.left), height(root.right)));
        }
    }

    // Insert method: used to create a binary search tree
    public static Node insert(Node root, int data) {
        if (root == null) {
            return new Node(data);
        } else {
            Node cur;
            if (data <= root.data) {
                cur = insert(root.left, data);
                root.left = cur;
            } else {
                cur = insert(root.right, data);
                root.right = cur;
            }
            return root;
        }
    }

    /**
     * Input:
     *            3
     *           / \
     *          2   5
     *         /   / \
     *        1   4   6
     *                 \
     *                  7
     *
     * Output:
     * 3
     */
    public static void main(String[] args) {
        Node root = null;
        int[] nodeValues = {3, 5, 2, 1, 4, 6, 7};
        for(int data: nodeValues)
            root = insert(root, data);

        // Get height
        int height = height(root);
        System.out.println(height);
    }
}