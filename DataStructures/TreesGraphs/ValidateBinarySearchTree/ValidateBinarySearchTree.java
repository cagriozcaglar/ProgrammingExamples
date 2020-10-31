/**
 Given a binary tree, determine if it is a valid binary search tree (BST).

 Assume a BST is defined as follows:

 The left subtree of a node contains only nodes with keys less than the node's key.
 The right subtree of a node contains only nodes with keys greater than the node's key.
 Both the left and right subtrees must also be binary search trees.

 Example 1:
 Input:
   2
  / \
 1   3
 Output: true

 Example 2:
     5
    / \
   1  4
  /    \
 3     6
 Output: false
 Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
 is 5 but its right child's value is 4.
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

public class ValidateBinarySearchTree {
    /**
     * Node range method: Starting from the root, define the range the node value should be in, and propagate the range
     * down the left and right subtree.
     * @param root
     * @return
     */
    public boolean isValidBST(TreeNode root) {
        return isValidBSThelper(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }

    /**
     *
     * Reason for long type for lower / upper, instead of int: When isValidBST(TreeNode root) is called with
     * [Integer.MIN_VALUE, Integer.MAX_VALUE] pair, if root is MAX_VALUE, which is the upper bound, upper bound
     * will be the same as root value, which will return false.
     *
     * @param root
     * @param lower: long
     * @param upper:long
     * @return
     */
    public boolean isValidBSThelper(TreeNode root, long lower, long upper) {
        // Error checks
        if(root == null) {
            return true;
        }

        // Check validity of current node
        if( !( (root.val > lower) &&
               (root.val < upper) ) ) {
            return false;
        }

        // Check validity of left subtree and right subtree
        // Left subtree: Update upper bound
        // Right subtree: Update lower bound
        if( !isValidBSThelper(root.left, lower, root.val) ||        // Left subtree check
            !isValidBSThelper(root.right, root.val, upper) ) {      // Right subtree check
            return false;
        }

        // If all tests above pass, the tree is a valid BST
        return true;
    }
}