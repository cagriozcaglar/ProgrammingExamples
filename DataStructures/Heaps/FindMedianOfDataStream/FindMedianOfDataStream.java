/**
 Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So
 the median is the mean of the two middle value.

 Examples:
 - [2,3,4], the median is 3
 - [2,3], the median is (2 + 3) / 2 = 2.5

 Design a data structure that supports the following two operations:
 1) void addNum(int num) - Add a integer number from the data stream to the data structure.
 2) double findMedian() - Return the median of all elements so far.

 For example:
 addNum(1)
 addNum(2)
 findMedian() -> 1.5
 addNum(3)
 findMedian() -> 2
 */

import java.util.*;

/**
 * Max heap comparator that sorts integers from highest to lowest
 */
class MaxHeapComparator implements Comparator<Integer> {
    public int compare(Integer number1, Integer number2){
        return number2-number1;
    }
}

/**
 * Min heap comparator that sorts integers from lowest to highest
 */
class MinHeapComparator implements Comparator<Integer> {
    public int compare(Integer number1, Integer number2){
        return number1-number2;
    }
}

/**
 * Use two heaps:
 * 1) Max heap: For the *lower half*, where we keep the *max* of lower half
 * 2) Min heap: For the *higher half*, where we keep the *min* of higher half
 * Example solution: https://www.programcreek.com/2015/01/leetcode-find-median-from-data-stream-java/
 */
public class FindMedianOfDataStream {
    // Heaps
    private PriorityQueue<Integer> maxHeap;
    private PriorityQueue<Integer> minHeap;

    /** initialize your data structure here. */
    public FindMedianOfDataStream() {
        maxHeap = new PriorityQueue<Integer>(new MaxHeapComparator());
        //maxHeap = new PriorityQueue<Integer>(Collections.reverseOrder());
        minHeap = new PriorityQueue<Integer>(new MinHeapComparator());
    }

    /**
     * Add element to the data structure
     *
     * Time complexity: O(log(n)) (Adding an element requires multiple element additions or removals from heaps, which are O(log(n)))
     * Note: Both insertion and extractMin / extractMax operations of heaps are O(log(n)) (Extraction is *not* O(1)).
     *
     * @param num
     */
    public void addNum(int num) {
        // Push to maxHeap first. If the size diff is > 1, push from maxHeap to minHeap
        // Invariant: 1 >= maxHeap.size() - minHeap.size() >= 0 .
        if( (maxHeap.size() == 0) || (maxHeap.size()>0 && maxHeap.peek() >= num) ) {
            maxHeap.offer(num);
        } else {
            minHeap.offer(num);
        }

        // Size balancing
        if(maxHeap.size() > minHeap.size()+1){
            minHeap.offer(maxHeap.poll());
        } else if(maxHeap.size() < minHeap.size()) {
            maxHeap.offer(minHeap.poll());
        }
        // System.out.println("Size of maxHeap: " + maxHeap.size());
        // System.out.println("Size of minHeap: " + minHeap.size());
    }

    /**
     * Find median
     *
     * Time complexity: O(1)
     * @return
     */
    public double findMedian() {
        // If there are even number of items, maxHeap and minHeap size are equal. Median is the average of two mid-elements.
        if(maxHeap.size() == minHeap.size()){
            // CAREFUL: Use peek() instead of poll() to get the values from heaps. You need the value, but you should not remove value from the heap.
            // CAREFUL: Cast the maxHeap.size()+minHeap.size() sum to double, before taking the average.
            return (double)(maxHeap.peek() + minHeap.peek()) / 2;
        } // If there are odd number of items, median is the middle element, which is the root of maxHeap, because maxHeap.size() = minHeap.size()+1 in this case.
        else {
            return maxHeap.peek();
        }
    }

    public static void main(String[] args) {
        FindMedianOfDataStream medianFinder = new FindMedianOfDataStream();
        medianFinder.addNum(2);
        System.out.println(medianFinder.findMedian()); // Median: 2
        medianFinder.addNum(3);
        System.out.println(medianFinder.findMedian()); // Median: 2.5
        medianFinder.addNum(4);
        System.out.println(medianFinder.findMedian()); // Median: 3
        medianFinder.addNum(5);
        System.out.println(medianFinder.findMedian()); // Median: 3.5
        medianFinder.addNum(6);
        System.out.println(medianFinder.findMedian()); // Median: 4
    }
}