/**
 Given a binary tree, determine if it is height-balanced.
 For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of
 every node never differ by more than 1.
 */

/**
 * Definition for a binary tree node.
 */
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

public class BalancedBinaryTree {
    /**
     * Solution 1: Check depth difference of left / right nodes at every node, and make recursive calls to isBalanced()
     * on left / right subtree.
     * Solution is similar to this one: http://www.geeksforgeeks.org/how-to-determine-if-a-binary-tree-is-balanced/
     *
     * Time complexity: O(n^2), not very optimal. This is to be optimized.
     *
     * @param root
     * @return
     */
    public static boolean isBalanced1(TreeNode root) {
        if(root == null){
            return true;
        } else {
            return Math.abs( depth(root.left) - depth(root.right)) <= 1 &&
                    isBalanced1(root.left) &&
                    isBalanced1(root.right);
        }
    }

    /**
     * Helper function depth() for Solution 1
     * Return the depth of a tree
     * @param root
     * @return
     */
    public static int depth(TreeNode root) {
        return (root == null ? 0 : 1 + Math.max(depth(root.left), depth(root.right)));
    }

    /**
     * Solution 2: Min-depth / Max-depth approach
     * A tree is considered balanced when the difference between the min depth and max depth does not exceed 1.
     * Similar to this one: http://www.mytechinterviews.com/balanced-tree
     *
     * @param root
     * @return
     */
    public static boolean isBalanced2(TreeNode root) {
        return (maxDepth(root) - minDepth(root) <= 1);
    }

    /**
     * Helper function minDepth() for Solution 2
     * Return the minimum depth of leaf nodes in a tree
     * @param root
     * @return
     */
    public static int minDepth(TreeNode root) {
        return (root == null) ? 0 : (1 + Math.min(minDepth(root.left), minDepth(root.right)) );
    }

    /**
     * Helper function maxDepth() for Solution 2
     * Return the maximum depth of leaf nodes in a tree
     * @param root
     * @return
     */
    public static int maxDepth(TreeNode root) {
        return (root == null) ? 0 : (1 + Math.max(maxDepth(root.left), maxDepth(root.right)) );
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
    public static TreeNode contractBinaryTree1(){
        // Nodes
        TreeNode root = new TreeNode(5);
        TreeNode rootLeft = new TreeNode(2);
        TreeNode rootRight = new TreeNode(4);
        TreeNode rootLeftLeft = new TreeNode(20);
        TreeNode rootLeftRight = new TreeNode(1);
        // TreeNode rootLeftRightRight = new TreeNode(9);
        // Edges
        root.left = rootLeft;
        root.right = rootRight;
        rootLeft.left = rootLeftLeft;
        rootLeft.right = rootLeftRight;
        // rootLeftRight.right = rootLeftRightRight;

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
    public static TreeNode contractBinaryTree2(){
        // Nodes
        TreeNode root = new TreeNode(12);
        TreeNode rootLeft = new TreeNode(4);
        TreeNode rootLeftLeft = new TreeNode(3);
        TreeNode rootLeftLeftLeft = new TreeNode(1);
        TreeNode rootLeftLeftLeftLeft  = new TreeNode(20);
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
    public static TreeNode contractBinaryTree3(){
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
     * Run a test case with all solutions
     * @param node
     */
    public static void runTestCase(TreeNode node){
        System.out.println("isBalanced1(node):" + isBalanced1(node));
        System.out.println("isBalanced2(node):" + isBalanced2(node));
        System.out.println();
    }

    public static void main(String[] args){
        // Construct example binary trees
        TreeNode bt1 = contractBinaryTree1();
        TreeNode bt2 = contractBinaryTree2();
        TreeNode bt3 = contractBinaryTree3();

        // Test 1: Binary tree 1: True
        /**
         *         5
         *       /  \
         *      2   4
         *     / \
         *    20  1
         */
        runTestCase(bt1);

        // Test 2: Binary tree 2: False
        /**
         *           12
         *          /  \
         *        4     5
         *       /
         *      3
         *     /
         *    1
         *   /
         *  20
         */
        runTestCase(bt2);

        // Test 3: Binary tree 3: False
        /**
         *           1
         *          / \
         *        2    5
         *       /      \
         *      3        6
         *     /          \
         *    4            7
         */
        runTestCase(bt3);
    }
}