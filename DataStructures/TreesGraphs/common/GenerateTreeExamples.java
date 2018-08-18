/**
 * Definition for a binary tree node.
 */
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x) { val = x; }
}

public static class GenerateTreeExamples {
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
}