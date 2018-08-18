/**
 Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).
 If two nodes are in the same row and column, the order should be from left to right.

 Examples:
 1) Given binary tree [3,9,20,null,null,15,7],
       3
      /\
     /  \
     9  20
       / \
      /   \
     15   7
 return its vertical order traversal as:
 [
 [9],
 [3,15],
 [20],
 [7]
 ]

 2) Given binary tree [3,9,8,4,0,1,7],
         3
        / \
       /   \
      9     8
     /\    / \
    / \   /  \
   4  0  1   7
 return its vertical order traversal as:
 [
 [4],
 [9],
 [3,0,1],
 [8],
 [7]
 ]

 3) Given binary tree [3,9,8,4,0,1,7,null,null,null,2,5] (0's right child is 2 and 1's left child is 5),


       3
      / \
     /   \
    9     8
   /\     /\
  /  \   /  \
 4   0  1   7
      \/
     /\
    5  2
 return its vertical order traversal as:
 [
 [4],
 [9,5],
 [3,0,1],
 [8,2],
 [7]
 ]
 */

// Example solution: https://www.geeksforgeeks.org/print-binary-tree-vertical-order-set-2/
// Example solution: https://www.programcreek.com/2014/04/leetcode-binary-tree-vertical-order-traversal-java/

import java.util.*;

public class BinaryTreeVerticalOrderTraversal {

    private static class Node {
        int key;
        Node left;
        Node right;

        Node(int data){
            key = data;
            left = null;
            right = null;
        }
    }

    /**
     *
     * @param root
     * @return
     */
    public static List<List<Integer>> printBinaryTreeVerticalOrderTraversal(Node root) {
        // Horizontal distance of root node is 0
        int distance = 0;

        // TreeMap to store Horizontal Distance -> Queue<Node> mapping
        // 1) We need TreeMap, because we need the keys, which is the horizontal distance, to be sorted, for iteration later
        // 2) We need Queue, because for each horizontal distance, we need the first node added in an upper level to be dequeued
        // first later as well (FIFO)
        TreeMap<Integer, Queue<Integer>> distanceNodeQueueMap = new TreeMap<Integer, Queue<Integer>>();

        // Populate vertical order with distanceNodeQueueMap and distance
        populateBinaryTreeVerticalOrderTraversal(root, distance, distanceNodeQueueMap);

        // Print vertical order.
        List<List<Integer>> verticalOrder = new ArrayList<List<Integer>>();
        // NOTE: When iterating over TreeMap, use "distanceNodeQueueMap.entrySet()", not "distanceNodeQueueMap" only
        for(Map.Entry<Integer, Queue<Integer>> levelData : distanceNodeQueueMap.entrySet()) {
            List<Integer> valuesForCurrentLevel = new ArrayList<Integer>();
            for(Integer value: levelData.getValue()) {
                valuesForCurrentLevel.add(value);
            }
            verticalOrder.add(valuesForCurrentLevel);
        }

        return verticalOrder;
    }

    /**
     *
     * @param root
     * @param distance
     * @param distanceNodeQueueMap
     */
    public static void populateBinaryTreeVerticalOrderTraversal(Node root, int distance, TreeMap<Integer, Queue<Integer>> distanceNodeQueueMap) {
        // Error checks
        if(root == null) {
            return;
        }

        // Add value of the current node to map
        if(!distanceNodeQueueMap.containsKey(distance)) {
            // NOTE: Queue is abstract, instantiate the queue using a LinkedList instead
            distanceNodeQueueMap.put(distance, new LinkedList<Integer>());
        }
        distanceNodeQueueMap.get(distance).add(root.key);

        // Branch left: Decrement distance
        populateBinaryTreeVerticalOrderTraversal(root.left, distance-1, distanceNodeQueueMap);

        // Branch right: Increment distance
        populateBinaryTreeVerticalOrderTraversal(root.right, distance+1, distanceNodeQueueMap);
    }

    /**
     * Create a binary tree 1:
     *         5
     *       /  \
     *      2   4
     *     / \
     *    20  1
     * @return root node of example binary tree
     */
    public static Node generateBinaryTree1() {
        // Nodes
        Node root = new Node(5);
        Node rootLeft = new Node(2);
        Node rootRight = new Node(4);
        Node rootLeftLeft = new Node(20);
        Node rootLeftRight = new Node(1);
        // Edges
        root.left = rootLeft;
        root.right = rootRight;
        rootLeft.left = rootLeftLeft;
        rootLeft.right = rootLeftRight;

        // Return root
        return root;
    }

    /**
     * Create a binary tree 2:
     *           12
     *          /  \
     *        4     5
     *       /
     *      3
     *     /
     *    1
     *   /
     *  20
     * @return root node of example binary tree
     */
    public static Node generateBinaryTree2() {
        // Nodes
        Node root = new Node(12);
        Node rootLeft = new Node(4);
        Node rootLeftLeft = new Node(3);
        Node rootLeftLeftLeft = new Node(1);
        Node rootLeftLeftLeftLeft = new Node(20);
        Node rootRight = new Node(5);
        // Edges
        root.left = rootLeft;
        rootLeft.left = rootLeftLeft;
        rootLeftLeft.left = rootLeftLeftLeft;
        rootLeftLeftLeft.left = rootLeftLeftLeftLeft;
        root.right = rootRight;

        // Return root
        return root;
    }

    /**
     * Create a binary tree 3:
     *           1
     *          / \
     *        2    5
     *       /      \
     *      3        6
     *     /          \
     *    4            7
     * @return root node of example binary tree
     */
    public static Node generateBinaryTree3() {
        // Nodes
        Node root = new Node(1);
        Node rootLeft = new Node(2);
        Node rootLeftLeft = new Node(3);
        Node rootLeftLeftLeft = new Node(4);
        Node rootRight = new Node(5);
        Node rootRightRight = new Node(6);
        Node rootRightRightRight = new Node(7);
        // Edges
        root.left = rootLeft;
        rootLeft.left = rootLeftLeft;
        rootLeftLeft.left = rootLeftLeftLeft;
        root.right = rootRight;
        rootRight.right = rootRightRight;
        rootRightRight.right = rootRightRightRight;

        // Return root
        return root;
    }

    public static void runTestCase(Node node) {
        // Print multi-level nested list
        // NOTE: Use Arrays.deepToString() to print multi-level arrays or lists or collections, instead of Arrays.toString()
        System.out.println(Arrays.deepToString(printBinaryTreeVerticalOrderTraversal(node).toArray()));
    }

    public static void main(String[] args) {
        // Test 1
        Node node1 = generateBinaryTree1();
        runTestCase(node1);

        // Test 2
        Node node2 = generateBinaryTree2();
        runTestCase(node2);

        // Test 3
        Node node3 = generateBinaryTree3();
        runTestCase(node3);
    }
}
