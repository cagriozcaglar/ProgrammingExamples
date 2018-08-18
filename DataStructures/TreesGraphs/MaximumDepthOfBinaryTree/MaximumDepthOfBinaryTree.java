/**
 Given a binary tree, find its maximum depth.
 The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
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

class MaximumDepthOfBinaryTree {
    /**
     * Solution: Recursive calls to maxDepth method with Depth-first search
     * @param root
     * @return
     */
    public static int maxDepth(TreeNode root) {
        // If root is null, depth is 0. Otherwise, include the node (1), and max of depth of left and righ subtree.
        return (root == null) ? 0 : 1+Math.max(maxDepth(root.left),maxDepth(root.right));
    }

    /**
     * Create an example BST:
     *         5
     *       /  \
     *     10   15
     *     / \
     *    8  6
     *        \
     *        7
     * @return root node of example binary tree
     */
    public static TreeNode contractBinaryTree1(){
        // Nodes
        TreeNode root = new TreeNode(5);
        TreeNode rootLeft = new TreeNode(10);
        TreeNode rootRight = new TreeNode(15);
        TreeNode rootLeftLeft = new TreeNode(8);
        TreeNode rootLeftRight = new TreeNode(6);
        TreeNode rootLeftRightRight = new TreeNode(7);
        // Connections
        root.left = rootLeft;
        root.right = rootRight;
        rootLeft.left = rootLeftLeft;
        rootLeft.right = rootLeftRight;
        rootLeftRight.right = rootLeftRightRight;

        // Return root
        return root;
    }

    /**
     * Create a binary tree:
     *           12
     *          /  \
     *        10   20
     *       /
     *     34
     *     /
     *   41
     *   /
     * -14
     * @return root node of example binary tree
     */
    public static TreeNode contractBinaryTree2(){
        // Nodes
        TreeNode root = new TreeNode(12);
        TreeNode rootLeft = new TreeNode(10);
        TreeNode rootLeftLeft = new TreeNode(34);
        TreeNode rootLeftLeftLeft = new TreeNode(41);
        TreeNode rootLeftLeftLeftLeft  = new TreeNode(-14);
        TreeNode rootRight = new TreeNode(20);
        // Connections
        root.left = rootLeft;
        rootLeft.left = rootLeftLeft;
        rootLeftLeft.left = rootLeftLeftLeft;
        rootLeftLeftLeft.left = rootLeftLeftLeftLeft;
        root.right = rootRight;

        // Return root
        return root;
    }

    public static void main(String[] args) {
        // Test 1: Depth: 4
        /**
         *         5
         *       /  \
         *     10   15
         *     / \
         *    8  6
         *        \
         *        7
         */
        TreeNode binaryTree1 = contractBinaryTree1();
        System.out.println("Maximum depth of this tree is: " + maxDepth(binaryTree1));

        // Test 2
        /**
         *           12
         *          /  \
         *        10   20
         *       /
         *     34
         *     /
         *   41
         *   /
         * -14
         */
        TreeNode binaryTree2 = contractBinaryTree2();
        System.out.println("Maximum depth of this tree is: " + maxDepth(binaryTree2));
    }
}