/**
 Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
 According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

 Example:
 Based on the BST below:
 1) The lowest common ancestor (LCA) of nodes 2 and 8 is 6.
 2) Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.

           _______6______
          /              \
      ___2__          ___8__
     /      \        /      \
    0       4       7       9
           / \
          3   5
 */


class Node {
    int val;
    Node left;
    Node right;
    // Constructor
    Node(int val){
        this.val = val;
    }
}

public class LowestCommonAncestorOfBinarySearchTree {

    /**
     *
     * @param root
     * @param node1
     * @param node2
     * @return
     */
    public static Node lcaOfBst1(Node root, Node node1, Node node2) {
        if(root == null || node1 == null || node2 == null) {
            return null;
        }

        // Difference values: Between root and node1 / node1
        int diff1 = root.val - node1.val;
        int diff2 = root.val - node2.val;

        // Four cases:
        // 1. Root is greater than both values, branch left
        if(diff1 > 0 && diff2 > 0) {
            return lcaOfBst1(root.left, node1, node2);
        } // 2. Root is smaller than both values, branch right
        else if(diff1 < 0 && diff2 < 0) {
            return lcaOfBst1(root.right, node1, node2);
        } // 3. Root is between two values (smaller and larger, not equal), return root
        else if(diff1 * diff2 < 0) {
            return root;
        } // 4. Root value is equal to one of the left / right values, in which case one of the nodes is LCA. Return root.
        else if(root.val == node1.val || root.val == node2.val) {
            return root;
        }
        // NOTE: A value outside if/else block has to be returned, otherwise this will return an error saying 'missing
        // return statement'. In this case, if conditions in if/else block do not match, there is no LCA, and return null.
        return null;
    }

    /**
     *
     * @param root
     * @param node1
     * @param node2
     * @return
     */
    public static Node lcaOfBst2(Node root, Node node1, Node node2){
        if(root == null || node1.equals(root) || node2.equals(root)) {
            return root;
        }

        Node left = lcaOfBst1(root.left, node1, node2);
        Node right = lcaOfBst1(root.right, node1, node2);

        if(left != null && right != null) {
            return root;
        }
        if(left != null){
            return left;
        } else {
            return right;
        }
    }

    /**
     *
     * @param root
     * @param node1
     * @param node2
     * @return
     */
    public static Node lcaOfBst3(Node root, Node node1, Node node2) {
        if(root == null || node1 == null || node2 == null){
            return null;
        }
        boolean node1Smaller = node1.val <= root.val;
        boolean node2Smaller = node2.val <= root.val;

        if(node1Smaller != node2Smaller){
            return root;
        } else {
            if(node1Smaller){
                return lcaOfBst2(root.left, node1, node2);
            } else {
                return lcaOfBst2(root.right, node1, node2);
            }
        }
        // NOTE: At this point, a return statement is not required. (Find out why.)
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

    /**
     * Run a test case with all solutions
     * @param root
     * @param node1
     * @param node2
     */
    public static void runTestCase(Node root, Node node1, Node node2){
        // Solution 1
        Node lca1 = lcaOfBst1(root, node1, node2);
        System.out.println("Lowest common ancestor of " + node1.val + " and " + node2.val + ": " + lca1.val);

        // Solution 2
        Node lca2 = lcaOfBst2(root, node1, node2);
        System.out.println("Lowest common ancestor of " + node1.val + " and " + node2.val + ": " + lca2.val);

        // Solution 3
        Node lca3 = lcaOfBst3(root, node1, node2);
        System.out.println("Lowest common ancestor of " + node1.val + " and " + node2.val + ": " + lca3.val);

        System.out.println();
    }

    public static void main(String[] args){
        // Create example BST
        Node bst1 = createExampleBst1();
        /**
         *        10
         *       /  \
         *      5   15
         *     / \
         *    3  7
         *      / \
         *     6  8
         */

        // Test 1: LCA of 3 and 8 => 5
        Node node1 = bst1.left.left;
        Node node2 = bst1.left.right.right;
        runTestCase(bst1, node1, node2);

        // Test 2: LCA of 7 and 8 => 7
        node1 = bst1.left.right;
        node2 = bst1.left.right.right;
        runTestCase(bst1, node1, node2);

        // Test 3: LCA of 6 and 15 => 10
        node1 = bst1.left.right.left;
        node2 = bst1.right;
        runTestCase(bst1, node1, node2);
    }
}
