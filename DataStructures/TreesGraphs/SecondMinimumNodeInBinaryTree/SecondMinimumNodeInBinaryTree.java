/**
 Given a non-empty special binary tree consisting of nodes with the non-negative value, where each node in this tree has
 exactly two or zero sub-node. If the node has two sub-nodes, then this node's value is the smaller value among its two sub-nodes.
 Given such a binary tree, you need to output the second minimum value in the set made of all the nodes' value in the whole tree.
 If no such second minimum value exists, output -1 instead.

 Example 1:
 Input:
     2
    / \
   2  5
  / \
 5  7
 Output: 5
 Explanation: The smallest value is 2, the second smallest value is 5.

 Example 2:
 Input:
   2
  / \
 2  2
 Output: -1
 Explanation: The smallest value is 2, but there isn't any second smallest value.
 */

package TreesGraphs.SecondMinimumNodeInBinaryTree;

/**
 * Definition for a binary tree node.
 */
class TreeNode {
     int val;
     TreeNode left;
     TreeNode right;
     TreeNode(int x) {
         val = x;
     }
}

// import TreesGraphs.PreorderTraversal.

class SecondMinimumNodeInBinaryTree {

    /**
     * Solution 1: Efficient
     *
     */
    public static int findSecondMinimumValue1(TreeNode root){
        // If root is null, there is no second minimum, return -1
        if(root == null){
            return -1;
        }

        // Find second minimum
        int res = findSecondMinimumHelper(root, root.val);

        // If second minimum is equal to root value, return -1, otherwise, return second minimum value.
        return (root.val == res) ? -1 : res;
    }

    /**
     * Helper for Solution 1
     */
    private static int findSecondMinimumHelper(TreeNode root, int prev){
        // If root is null, return the value of previous value, which is the the closest number we have so far
        if(root == null){
            return prev;
        }
        // If root value is different than prev (which means larger), return prev
        if(root.val != prev){
            return root.val;
        }

        // Get the second minimum of left and right trees
        int leftValue = findSecondMinimumHelper(root.left, root.val);
        int rightValue = findSecondMinimumHelper(root.right, root.val);

        // Based on leftValue and rightValue, return the second minimum value.
        if( (leftValue == root.val) && (rightValue == root.val) ){
            return prev;
        } else if(leftValue == root.val){
            return rightValue;
        } else if(rightValue == root.val){
            return leftValue;
        } else{
            return Math.min(leftValue, rightValue);
        }
    }

    /**
     * Solution 2: Inefficient, my way.
     * findSecondMinimumValue(TreeNode root)    .... Main caller
     *             |
     * findMinGreaterThanRoot(TreeNode rootLeft, TreeNode rootRight, int minValue)   .... Recursive function
     *             |
     * secondMinimumSelector(int leftVal, int rightVal, int minVal)   ... Selects second minimum value given three values
     */

    /**
     * Main caller
     * @param root
     * @return
     */
    public static int findSecondMinimumValue2(TreeNode root) {
        // If root is null, return -1. Otherwise, call find second minimum
        if(root == null){
            return -1;
        } else{
            // Find and return second minumum value
            int secondMinimum = findMinGreaterThanRoot(root.left, root.right, root.val);
            return secondMinimum;
        }
    }

    /**
     * Find minimum value greater than root, given left node, right node, and minimum value
     * @param rootLeft
     * @param rootRight
     * @param minValue
     * @return
     */
    public static int findMinGreaterThanRoot(TreeNode rootLeft, TreeNode rootRight, int minValue) {
        // If rootLeft is null (which also means rootRight is also null), return -1
        if(rootLeft == null){
            return -1;
        } else {  // Otherwise, check rootLeft and rootRight values, and find second minimum value
            // If both value rootLeft and rootRight are greater than minValue, return the minimum of them as the 2nd min.
            if(rootLeft.val > minValue && rootRight.val > minValue){
                return Math.min(rootLeft.val, rootRight.val);
            } // If rootLeft > minValue but rootRight is not, call secondMinimumSelector() on rootLeft.val and recursive call
              // findMinGreaterThanRoot(rootRight.left, rootRight.right, minValue)
            else if(rootLeft.val > minValue){
                return secondMinimumSelector(rootLeft.val, findMinGreaterThanRoot(rootRight.left, rootRight.right, minValue), minValue);
            } // If rootRight > minValue but rootLeft is not, call secondMinimumSelector() on recursive call
              // findMinGreaterThanRoot(rootLeft.left, rootLeft.right, minValue) and rootRight.val.
            else if(rootRight.val > minValue){
                return secondMinimumSelector(findMinGreaterThanRoot(rootLeft.left, rootLeft.right, minValue), rootRight.val, minValue);
            } // If rootLeft.val and rootRight.val are both not greater than minValue, call secondMinimumSelector() on recursive call
              // findMinGreaterThanRoot(rootLeft.left, rootLeft.right, minValue) and
              // findMinGreaterThanRoot(rootRight.left, rootRight.right, minValue)
            else{
                return secondMinimumSelector(findMinGreaterThanRoot(rootLeft.left, rootLeft.right, minValue), findMinGreaterThanRoot(rootRight.left, rootRight.right, minValue), minValue);
            }
        }
    }

    /**
     * Select second minimum value given left value, right value, and minimum value
     * @param leftVal
     * @param rightVal
     * @param minVal
     * @return
     */
    public static int secondMinimumSelector(int leftVal, int rightVal, int minVal) {
        if(leftVal > minVal && rightVal > minVal) {
            return Math.min(leftVal, rightVal);
        } else if(leftVal > minVal) {
            return leftVal;
        } else if(rightVal > minVal) {
            return rightVal;
        } else {
            return -1;
        }
    }

    /**
     * Main runner of all tests
     * @param root
     */
    public static void testRunner(TreeNode root){
        System.out.println("findSecondMinimumValue1 returns: " + findSecondMinimumValue1(root));
        System.out.println("findSecondMinimumValue2 returns: " + findSecondMinimumValue2(root));
    }

    public static void main(String[] args){
        // Test 1
        /*
                 2
                / \
               2   5
                  / \
                 5   7
        */
        TreeNode rootNode = new TreeNode(2);
        rootNode.left = new TreeNode(2);
        TreeNode rootRight = new TreeNode(5);
        rootNode.right = rootRight;
        rootRight.left = new TreeNode(5);
        rootRight.right = new TreeNode(7);

        testRunner(rootNode);


        // Test 2: No second minimum available
        /*
                 2
                / \
               2   2
        */
        TreeNode rootNode2 = new TreeNode(2);
        rootNode2.left = new TreeNode(2);
        rootNode2.right = new TreeNode(2);

        testRunner(rootNode2);

        // Test 3: Hard test case. Left tree is deep and returns 2, right tree returns 5. Minimum is 2.
        /*
                     1
                   /   \
                 1       5
               /   \
             1       1
           /   \    / \
          1     1  1   2
        */
        TreeNode rootNode3 = new TreeNode(1);
        TreeNode rootLeftNode = new TreeNode(1);
        TreeNode rootRightNode = new TreeNode(5);
        rootNode3.left = rootLeftNode;
        rootNode3.right = rootRightNode;
        TreeNode rootLeftLeftNode = new TreeNode(1);
        TreeNode rootLeftRightNode = new TreeNode(1);
        rootLeftNode.left = rootLeftLeftNode;
        rootLeftNode.right = rootLeftRightNode;
        rootLeftLeftNode.left = new TreeNode(1);
        rootLeftLeftNode.right = new TreeNode(1);
        rootLeftRightNode.left = new TreeNode(1);
        rootLeftRightNode.right = new TreeNode(2);

        testRunner(rootNode3);

    }
}