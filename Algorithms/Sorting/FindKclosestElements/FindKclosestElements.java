/**
 Given a sorted array, two integers k and x, find the k closest elements to x in the array.
 The result should also be sorted in ascending order. If there is a tie, the smaller elements are always preferred.

 Example 1:
  Input: [1,2,3,4,5], k=4, x=3
  Output: [1,2,3,4]
 Example 2:
  Input: [1,2,3,4,5], k=4, x=-1
  Output: [1,2,3,4]

 Note:
  1. The value k is positive and will always be smaller than the length of the sorted array.
  2. Length of the given array is positive and will not exceed 10^4.
  3. Absolute value of elements in the array and x will not exceed 10^4.
 */

/*******************
 * NOTE: DO NOT SUBMIT UNTIL YOU GO OVER OPTIMAL SOLUTIONS: O(n) and O(log(n)).
 * https://leetcode.com/problems/find-k-closest-elements/discuss/
 *******************/

import java.util.*;

class FindKclosestElements {
    /**
     * Find closest K elements
     * 1) Sort the array by an anonymous comparator which sorts the array by distance to x.
     * 2) Take first k values of the sorted array
     * 3) Sort the k-length array by their value, and return.
     * @param arr
     * @param k
     * @param x
     * @return
     */
    public static List<Integer> findClosestElements(List<Integer> arr, int k, int x) {

        // Sort the list of integers by distance to x, using an anonymous comparator
        Collections.sort(arr, new Comparator<Integer>(){
            public int compare(Integer value1, Integer value2){
                int distanceValue1 = Math.abs(value1 - x);
                int distanceValue2 = Math.abs(value2 - x);
                return (distanceValue1 - distanceValue2 < 0) ? -1 :
                        ((distanceValue1 - distanceValue2 > 0) ? 1 : ((value1 < value2) ? -1 : 1));
            }
        }); // end of sort operation

        // Get the first k elements in the list of integers sorted by distance to x.
        // Note: The values are not sorted by their values, they are sorted by their distance to x.
        List<Integer> arrClosestKelements = arr.subList(0,k);

        // Now, sort the list of integers by their values. There is no need for a new
        // anonymous Comparator, as the default comparator will sort integers by value.
        Collections.sort(arrClosestKelements);

        // Return the list of k closest elements, sorted by their values.
        return arrClosestKelements;
    }

    /**
     * Find closest K elements - optimized with shorter sorting
     * 1) Sort the array by an anonymous comparator which sorts the array by distance to x.
     * 2) Take first k values of the sorted array
     * 3) Sort the k-length array by their value, and return.
     * @param arr
     * @param k
     * @param x
     * @return
     */
    public static List<Integer> findClosestElementsOptimized(List<Integer> arr, int k, int x) {
        Collections.sort(arr, (a,b) -> a==b ? (a-b) : (Math.abs(a-x) - Math.abs(b-x)) );
        arr = arr.subList(0, k);
        Collections.sort(arr);
        return arr;
    }

    public static void printContents(List<Integer> integerList){
        for(Integer value: integerList){
            System.out.print(value + " ");
        }
        System.out.println();
    }

    public static void main(String[] args){
        // Timers for profiling
        long startTime;
        long endTime;

        List<Integer> integerList = Arrays.asList(1, 2, 3, 4, 5);
        int k = 4;
        int x = 3;

        // Method 1
        startTime = System.nanoTime();
        List<Integer> sortedIntegerList = findClosestElements(integerList, k, x);
        endTime = System.nanoTime();
        printContents(sortedIntegerList);
        System.out.println("Time for Method 1: " + (endTime - startTime));

        // Method 2
        startTime = System.nanoTime();
        sortedIntegerList = findClosestElementsOptimized(integerList, k, x);
        endTime = System.nanoTime();
        printContents(sortedIntegerList);
        System.out.println("Time for Method 2: " + (endTime - startTime));

        List<Integer> integerList2 = new ArrayList<Integer>();
        for(int i = 1; i <= 1000000; i++){
            integerList2.add(i);
        }
        k = 4000;
        x = 3;

        // Method 1
        startTime = System.nanoTime();
        List<Integer> sortedIntegerList2 = findClosestElements(integerList2, k, x);
        endTime = System.nanoTime();
        printContents(sortedIntegerList2);
        System.out.println("Time for Method 1: " + (endTime - startTime));

        // Method 2
        startTime = System.nanoTime();
        sortedIntegerList2 = findClosestElementsOptimized(integerList2, k, x);
        endTime = System.nanoTime();
        printContents(sortedIntegerList2);
        System.out.println("Time for Method 2: " + (endTime - startTime));

    }
}