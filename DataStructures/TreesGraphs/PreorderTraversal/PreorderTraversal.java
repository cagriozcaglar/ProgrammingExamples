/**
 Given a binary tree, return the preorder traversal of its nodes' values.

 For example:
 Given binary tree [1,null,2,3],
 1
  \
   2
  /
 3
 return [1,2,3].

 Note: Recursive solution is trivial, could you do it iteratively?
 */

/**
 * Example links: https://leetcode.com/problems/binary-tree-preorder-traversal/discuss/45468/3-Different-Solutions
 */

// package TreesGraphs.PreorderTraversal;

import java.util.*; // For Stack, ArrayList
// import DataStructures.TreesGraphs.common.GenerateTreeExamples;
// import TreesGraphs.common.GenerateTreeExamples;
// import common.GenerateTreeExamples;

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

public class PreorderTraversal {
    /**
     * 1. Iterative Preorder Traversal.
     * DFS Preorder traversal using stack
     * Note: Preorder is CLR (Current, Left, Right). When pushing to stack, current is added to the list first, then
     * right node is added to stack first (to be accessed last), and left node is added to stack last (to be accessed first)
     * @param root
     * @return
     */
    public static List<Integer> preorderTraversalIterative(TreeNode root) {
        // Stack to use to implement iterative DFS - preorder traversal
        Stack<TreeNode> nodeStack = new Stack<TreeNode>();
        // Traversal result list to be returned
        List<Integer> traversalResult = new ArrayList<Integer>();

        // Push first node to stack
        nodeStack.push(root);

        // Iterate over the nodes until nodeStack is empty
        while(!nodeStack.isEmpty()) {
            // Get the top node in the stack, then pop from stack
            TreeNode current = nodeStack.peek();
            nodeStack.pop();

            // If current node is not null, push unvisited neighbours.
            // Order matters here: Preorder is CLR (current, left, right). We first print current node, then we push right node (last to be examined), then left node (to be examined before right node)
            if(current != null) {
                // C: Add current node to traversal list
                traversalResult.add(current.val);

                // R: Push right node to stack (first in, last out)
                nodeStack.push(current.right);
                // L: Push left node to stack (last in, first out)
                nodeStack.push(current.left);

            }
        }
        return traversalResult;
    }

    /**
     * 2. Recursive Preorder Traversal: Single function, without helper
     * @param root
     * @return
     */
    public static List<Integer> preorderTraversalRecursiveWithoutHelper(TreeNode root) {
        // Traversal result list to be returned
        List<Integer> traversalResult = new ArrayList<Integer>();

        // If current node is null, return the empty traversal list
        if(root == null) { return traversalResult; }

        // C: Add current node
        traversalResult.add(root.val);
        // L: Add / append the traversal list of left node to the existing traversal list
        traversalResult.addAll(preorderTraversalRecursiveWithoutHelper(root.left));
        // R: Add / append the traversal list of right node to the existing traversal list
        traversalResult.addAll(preorderTraversalRecursiveWithoutHelper(root.right));

        return traversalResult;
    }

    /**
     * 3. Recursive Preorder Traversal: With helper function
     * Creates an empty traversal list, passes it to helper function as a parameter, and helper populates the traversal
     * result list. Main work is done in the helper function below
     * @param root
     * @return
     */
    public static List<Integer> preorderTraversalRecursiveWithHelper(TreeNode root) {
        List<Integer> traversalResult = new ArrayList<Integer>();
        traversalHelper(root, traversalResult);
        return traversalResult;
    }

    /**
     * Helper function for "3. Recursive Preorder Traversal" above.
     * Add elements to result list, first current node, then calling left node with result list, then calling right node
     * with result list. This function returns void, it only modifies the traversal list.
     */
    public static void traversalHelper(TreeNode root, List<Integer> traversalResult) {
        if(root == null) { return; }
        traversalResult.add(root.val);
        traversalHelper(root.left, traversalResult);
        traversalHelper(root.right, traversalResult);
    }

    //TODO: Move example binary tree generation code to a common library.
    /**
     * Create a binary tree 1:
     *         5
     *       /  \
     *      2   4
     *     / \
     *    20  1
     * @return root node of example binary tree
     */
    public static TreeNode generateBinaryTree1() {
        // Nodes
        TreeNode root = new TreeNode(5);
        TreeNode rootLeft = new TreeNode(2);
        TreeNode rootRight = new TreeNode(4);
        TreeNode rootLeftLeft = new TreeNode(20);
        TreeNode rootLeftRight = new TreeNode(1);
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
    public static TreeNode generateBinaryTree2() {
        // Nodes
        TreeNode root = new TreeNode(12);
        TreeNode rootLeft = new TreeNode(4);
        TreeNode rootLeftLeft = new TreeNode(3);
        TreeNode rootLeftLeftLeft = new TreeNode(1);
        TreeNode rootLeftLeftLeftLeft = new TreeNode(20);
        TreeNode rootRight = new TreeNode(5);
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
    public static TreeNode generateBinaryTree3() {
        // Nodes
        TreeNode root = new TreeNode(1);
        TreeNode rootLeft = new TreeNode(2);
        TreeNode rootLeftLeft = new TreeNode(3);
        TreeNode rootLeftLeftLeft = new TreeNode(4);
        TreeNode rootRight = new TreeNode(5);
        TreeNode rootRightRight = new TreeNode(6);
        TreeNode rootRightRightRight = new TreeNode(7);
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

    /**
     *
     * @param args
     */
    public static void runAllTests(TreeNode node) {
        // Method 1: Iterative Preorder Traversal
        List<Integer> preorderTraversalIterativeList = preorderTraversalIterative(node);
        System.out.println("Iterative Preorder Traversal: " + Arrays.toString(preorderTraversalIterativeList.toArray()));

        // Method 2: Recursive Preorder Traversal without helper
        List<Integer> preorderTraversalRecursiveWithoutHelperList = preorderTraversalRecursiveWithoutHelper(node);
        System.out.println("Recursive Preorder Traversal Without Helper: " + Arrays.toString(preorderTraversalRecursiveWithoutHelperList.toArray()));

        // Method 3: Recursive Preorder Traversal with helper
        List<Integer> preorderTraversalRecursiveWithHelperList = preorderTraversalRecursiveWithHelper(node);
        System.out.println("Recursive Preorder Traversal With Helper: " + Arrays.toString(preorderTraversalRecursiveWithHelperList.toArray()));

        System.out.println();
    }

    public static void main(String[] args) {
        // Binary tree 1
        TreeNode binaryTree1 = generateBinaryTree1();
        runAllTests(binaryTree1);

        // Binary tree 2
        TreeNode binaryTree2 = generateBinaryTree2();
        runAllTests(binaryTree2);

        // Binary tree 3
        TreeNode binaryTree3 = generateBinaryTree3();
        runAllTests(binaryTree3);
    }
}