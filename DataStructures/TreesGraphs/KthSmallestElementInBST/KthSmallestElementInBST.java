/**
 Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

 Note:
 You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

 Follow up:
 What if the BST is modified (insert/delete operations) often and you need to find the kth smallest frequently? How would you optimize the kthSmallest routine?

 */

import java.util.*;

class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    // Constructor
    TreeNode(int x) {
        val = x;
    }
}

public class KthSmallestElementInBST {
    /**
     * Solution 1: Using inorder traversal
     *
     * Time complexity: O(n)
     *
     * @param root
     * @param k
     * @return
     */
    public static int kthSmallestElementInBST(TreeNode root, int k) {
        List<Integer> orderedNodes = new ArrayList<Integer>();
        inorderTraversal(root, orderedNodes);
        return orderedNodes.get(k-1);
    }

    /**
     * Helper: Write inorder traversal of a tree into a list, which is passed as an argument
     * @param root
     * @param orderedNodes
     */
    public static void inorderTraversal(TreeNode root, List<Integer> orderedNodes) {
        if(root == null) {
            return;
        }
        inorderTraversal(root.left, orderedNodes);
        orderedNodes.add(root.val);
        inorderTraversal(root.right, orderedNodes);
    }

    /**
     * Solution 2: Using binary search on the number of nodes on the left of each subtree
     *
     * Time complexity: O(log(n)) for balanced BST, O(n) for degenerate BST (worst case)
     *
     * @param root
     * @param k
     * @return
     */
    public static int kthSmallesInBSTwithBinarySearch(TreeNode root, int k) {
        int count = countNodes(root.left);
        if(k <= count) {
            return kthSmallesInBSTwithBinarySearch(root.left, k);
        } else if(k > count+1) {
            return kthSmallesInBSTwithBinarySearch(root.right, k-1-count);
        }

        return root.val;
    }

    /**
     * Helper: Return the number of nodes in the tree rooted at the node
     * @param n
     * @return
     */
    public static int countNodes(TreeNode n) {
        if(n == null) {
            return 0;
        }
        return 1 + countNodes(n.left) + countNodes(n.right);
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
    public static TreeNode createExampleBst1(){
        TreeNode root = new TreeNode(10);
        TreeNode rootLeft = new TreeNode(5);
        TreeNode rootRight = new TreeNode(15);
        root.left = rootLeft;
        root.right = rootRight;

        TreeNode rootLeftLeft = new TreeNode(3);
        TreeNode rootLeftRight = new TreeNode(7);
        rootLeft.left = rootLeftLeft;
        rootLeft.right = rootLeftRight;

        rootLeftRight.left = new TreeNode(6);
        TreeNode rootLeftRightRight = new TreeNode(8);
        rootLeftRight.right = rootLeftRightRight;

        return root;
    }

    public static void runTestCase(TreeNode node, int k) {
        System.out.println(k + "-th smallest element of this tree is: " + kthSmallestElementInBST(node,k) );
        System.out.println(k + "-th smallest element of this tree is: " + kthSmallesInBSTwithBinarySearch(node,k) );
    }

    public static void main(String[] args) {
        TreeNode bst1 = createExampleBst1(); // [3,5,6,7,8,10,15]

        // Test 1
        runTestCase(bst1, 3);  // 6

        // Test 2
        runTestCase(bst1, 5);  // 8
    }
}