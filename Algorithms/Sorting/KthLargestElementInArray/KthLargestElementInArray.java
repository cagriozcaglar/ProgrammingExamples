/**
 Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the
 kth distinct element.

 Example 1:
 Input: [3,2,1,5,6,4] and k = 2
 Output: 5

 Example 2:
 Input: [3,2,3,1,2,4,5,5,6] and k = 4
 Output: 4

 Note:
 You may assume k is always valid, 1 ≤ k ≤ array's length.
 */

import java.util.*;

public class KthLargestElementInArray {

    /**
     * Quick select
     *
     * Example 1: https://www.programcreek.com/2014/05/leetcode-kth-largest-element-in-an-array-java/
     * Example 2: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/215.%20Kth%20Largest%20Element%20in%20an%20Array.java
     * Example 3: https://www.geeksforgeeks.org/kth-smallestlargest-element-unsorted-array/
     *
     * @param nums
     * @param k
     * @return
     */
    public static int kthLargestElementInArrayWithQuickSelect(int[] nums, int k) {
        // Error check
        if(nums == null || k < 1 || nums.length < k) {
            return 0;
        }

        return getKthLargest(nums, 0, nums.length-1, k);
    }

    public static int getKthLargest(int[] nums, int left, int right, int k) {
        while(true) {
            int position = partition(nums, left, right);
            if(position == k-1) {
                return nums[position];
            } else if(position < k-1) {
                left = position+1;
            } else {
                right = position-1;
            }
        }
    }

    /**
     * Partition the array around pivot, such that nums[i] < pivot for left indices, nums[i] > pivot for right indices
     * Return the end of left partition.
     *
     * @param nums
     * @param left
     * @param right
     * @return
     */
    public static int partition(int[] nums, int left, int right){
        int pivot = nums[randomIntegerInRange(left, right)];
        while(true) {
            // While nums[left] <= pivot, keep incrementing left
            while(left <= right && nums[left] <= pivot) {
                left++;
            }
            // While nums[right] > pivot, keep decrementing right
            while(left <= right && nums[right] > pivot) {
                right--;
            }
            // If left passed right pointer, return left-1 as the final position
            if(left > right) {
                return left-1;
            }
            // Here, both left and right pointers are on the incorrect side. nums[left] > pivot and nums[right] <= pivot.
            // Therefore, swap left and right pointers, so they are on the correst side.
            swap(nums, left, right);
        }
    }

    public static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public static int randomIntegerInRange(int left, int right) {
        return left + (int) (Math.random() * (right-left+1)); // [left, left] + [0,1)*(right-left+1) = [left,right+1) = [left,right]
    }

    /**
     * Use a *min-heap* of largest k numbers (Because we need the minimum of largest k numbers for comparison)
     * Add elements of the array to min-heap. As you add elements, if the size of the min-heap exceeds k, delete minimum,
     * which is the root of min-heap. This way, you keep a record of k largest elements.
     *
     * Similar solution in Program Creek: https://www.programcreek.com/2014/05/leetcode-kth-largest-element-in-an-array-java/
     * Similar solution: https://leetcode.com/problems/kth-largest-element-in-an-array/discuss/129703/Java-8-PriorityQueue
     *
     * Time complexity: O(n * log(k)) (n is the array size, k is the order of the element we want to find) (Explanation:
     *         For each of the n elements in the array, we insert it to min-heap, each of which takes O(log(k)) time.)
     * Space complexity: O(k) (For storing top k numbers in the heap)
     *
     * @param nums
     * @param k
     * @return
     */
    public static int kthLargestElementInArrayWithHeap(int[] nums, int k) {
        // Create a *min-heap* of largest k numbers (Because we need the minimum of largest k numbers for comparison)
        PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>( (x,y) -> y-x );
        // PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();  // This is also OK. Default is min-Heap order

        for(int num : nums) {
            minHeap.offer(num);

            // Check size, if > k, drop minimum of the heap
            if(minHeap.size() > k) {
                minHeap.poll();
            }
        }

        // Return the minimum element of the minHeap of size k, which is also the k-th largest element
        return minHeap.peek();
    }

    public static void runTestCase(int[] nums, int k) {
        System.out.println(k + "-th largest element of " + Arrays.toString(nums) + " using MinHeap is: " +
                           kthLargestElementInArrayWithHeap(nums,k) );
        System.out.println(k + "-th largest element of " + Arrays.toString(nums) + " using QuickSelect is: " +
                           kthLargestElementInArrayWithQuickSelect(nums,k) );
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        int[] nums = new int[]{3,2,1,5,6,4};
        int k = 5;
        runTestCase(nums, k);

        // Test 2
        int[] nums2 = new int[]{3,2,3,1,2,4,5,5,6};
        int k2 = 4;
        runTestCase(nums2, k2);
    }
}