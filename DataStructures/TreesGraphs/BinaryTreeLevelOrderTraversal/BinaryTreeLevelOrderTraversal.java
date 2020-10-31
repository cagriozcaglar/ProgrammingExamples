/**
 Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

 For example:
 Given binary tree [3,9,20,null,null,15,7],
     3
    / \
   9  20
  /    \
 15    7
 return its level order traversal as:
 [
   [3],
   [9,20],
   [15,7]
 ]
*/

import java.util.*;


/**
 * Definition for a binary tree node.
 */
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

public class BinaryTreeLevelOrderTraversal {
    /**
     * Solution 1: Using BFS with one queue, without helper function
     *
     * Running time: O(n), where n is the number of nodes in the tree rooted at root.
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrder(TreeNode root) {
        // Queue for implementing BFS
        Queue<TreeNode> nodeQueue = new LinkedList<TreeNode>();
        // Final list of list of nodes at each level
        List<List<Integer>> levelOrderTraversal = new LinkedList<List<Integer>>();

        // If root is null, no nodes in the tree, return the empty list of list of nodes for level-order traversal
        if(root == null) {
            return levelOrderTraversal;
        }

        // Enqueue initial node (root)
        nodeQueue.offer(root);
        while(!nodeQueue.isEmpty()) {
            // Get the fixed(!, important) size of the queue, which returns the node count in the current level.
            // Do not use queue.size() in the for loop later, as the size of queue will change.
            int nodeCountInLevel = nodeQueue.size(); // Number of nodes in the queue is fixed at this point, for the current level

            // Create Empty list of nodes for the current level
            List<Integer> nodesInLevel = new LinkedList<Integer>();

            // Iterate over all nodes in the current level, add them to nodesInLevel list, add their children to queue,
            // dequeue the node itself from the queue.
            for(int i=0; i < nodeCountInLevel; i++){
                // Get current node
                TreeNode currentNode = nodeQueue.peek();
                // Add left / right children of current node, if they are not null
                if(currentNode.left != null){
                    nodeQueue.offer(currentNode.left);
                }
                if(currentNode.right != null){
                    nodeQueue.offer(currentNode.right);
                }
                // Remove the current node from queue, and add it to final nodesInLevel list
                nodesInLevel.add(nodeQueue.poll().val);
            }
            // Add nodes in current level to final list of list of nodes
            levelOrderTraversal.add(nodesInLevel);
        }
        return levelOrderTraversal;
    }

    /**
     * Solution 2: BFS using a helper function which takes the following arguments: 1) List of list of nodes in levels,
     * 2) depth.
     *
     * Time complexity: O(n), where n is the number of nodes in the tree rooted at root.
     *
     * @param root
     * @return
     */
    public static List<List<Integer>> levelOrderWithHelper(TreeNode root) {
        // Initial empty list of list of nodes in all levels
        List<List<Integer>> nodesInAllLevels = new LinkedList<List<Integer>>();
        // Fill the list of nodes in levels
        buildLevelOrder(root, nodesInAllLevels, 0);
        return nodesInAllLevels;
    }

    /**
     * Helper function for Solution 2: Given root node, current state of list of list of nodes in levels, and depth,
     * recursively builds the list of list of nodes at each level.
     *
     * Time complexity: O(n), where n is the number of nodes in the tree rooted at root.
     *
     * @param root
     * @param nodesInAllLevels
     * @param depth
     */
    public static void buildLevelOrder(TreeNode root, List<List<Integer>> nodesInAllLevels, int depth) {
        if(root == null){
            return;
        }
        // If the depth is equal to number of levels in nodesInAllLevels list, it is time to push a new empty list of
        // nodes to nodesInAllLevels list for the current depth
        if(nodesInAllLevels.size() == depth){
            nodesInAllLevels.add(new LinkedList<Integer>());
        }

        // Add root to list of nodes for the current level
        nodesInAllLevels.get(depth).add(root.val);
        // Add nodes in left / right child
        buildLevelOrder(root.left, nodesInAllLevels, depth+1);
        buildLevelOrder(root.right, nodesInAllLevels, depth+1);
    }

    public static void runTestCases(TreeNode root) {
        List<List<Integer>> levelOrderTraversalMethod1 = levelOrder(root);
        System.out.println("Level order traversal using method 1: " + Arrays.toString(levelOrderTraversalMethod1.toArray()));
        List<List<Integer>> levelOrderTraversalMethod2 = levelOrderWithHelper(root);
        System.out.println("Level order traversal using method 2: " + Arrays.toString(levelOrderTraversalMethod2.toArray()));
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

    public static void main(String[] args){
        // Test 1
        TreeNode binaryTreeRoot1 = generateBinaryTree1();
        runTestCases(binaryTreeRoot1);

        // Test 2
        TreeNode binaryTreeRoot2 = generateBinaryTree2();
        runTestCases(binaryTreeRoot2);

        // Test 3
        TreeNode binaryTreeRoot3 = generateBinaryTree3();
        runTestCases(binaryTreeRoot3);
    }
}
