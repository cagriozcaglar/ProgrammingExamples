/**
 Given a collection of intervals, merge all overlapping intervals.

 Example 1:
 Input: [[1,3],[2,6],[8,10],[15,18]]
 Output: [[1,6],[8,10],[15,18]]
 Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

 Example 2:
 Input: [[1,4],[4,5]]
 Output: [[1,5]]
 Explanation: Intervals [1,4] and [4,5] are considerred overlapping.
 */

import java.util.*;

class Interval {
    int start;
    int end;
    Interval(int s, int e) {
        start = s;
        end = e;
    }

    /**
     * This is for printing Interval objects with [start, end] format (instead of printing)
     * @return formatted
     */
    public String toString() {
        return "[" + start + ", " + end + "]";
    }
}

public class MergeIntervals{
    /**
     * Algorithm:
     * 1. Sort intervals by starting position.
     * 2. Iterate over the intervals sorted by starting position:
     *   2.1. If current and previous intervals intersect (current.start <= prev.end), merge intervals by updating end
     *          => end = max(current.end, previous.end)
     *   2.2. If current and previous intervals do not intersect, add current interval with (start, end) pair to the list
     *       of merged intervals.
     * 3. At the end, add the final interval with (start, end) pair.
     *
     * Time complexity: O(nlog(n)) (Because of sorting step)
     * Space complexity: O(n) (Merged intervals list)
     *
     * Example solution: https://leetcode.com/problems/merge-intervals/solution/
     * Example solution 2: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/56.%20Merge%20Intervals.java
     * @param intervals
     * @return
     */
    public static List<Interval> mergeIntervals(List<Interval> intervals) {
        // Checks
        if (intervals.size() <= 1) {
            return intervals;
        }

        List<Interval> mergedIntervals = new ArrayList<>();

        // Sort intervals by starting position
        Collections.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval i1, Interval i2) {
                return i1.start - i2.start;
            }
        });

        // Iterate over sorted intervals, and merge if start_{i+1} <= end_i at any step
        // Initialize start and end to start / end of first interval after sorting
        // Do not add this interval to merged intervals yet, because it may be merged with neighbour intervals
        int start = intervals.get(0).start;
        int end = intervals.get(0).end;
        // Iterate over sorted intervals
        for (int i = 1; i < intervals.size(); i++) {
            Interval currentInterval = intervals.get(i);
            // If start of current interval is <= end of previous interval, two intervals intersect. Update end point, start remains the same.
            if (currentInterval.start <= end) {
                end = Math.max(end, currentInterval.end); // maximum of both previous and current endpoints
            } // If consecutive intervals do not overlap, then create the current merged interval, add to list of merged intervals
            else {
                mergedIntervals.add(new Interval(start, end));
                // Update start and end
                // NOTE: Start is only updated after we add a merged interval. When two consecutive intervals overlap, start is not updated
                start = currentInterval.start; // Start of added interval, to be used in next iteration as previous start
                end = currentInterval.end;     // End of added interval, to be used in next iteration as previous start
            }
        }

        // CAREFUL: Add the last interval with (start, end) pair.
        mergedIntervals.add(new Interval(start, end));
        return mergedIntervals;
    }

    public static void runTestCase(List<Interval> intervals) {
        System.out.println("After merging intervals in " + Arrays.toString(intervals.toArray()) +
                           " , the resulting merged interval list is: " + Arrays.toString(mergeIntervals(intervals).toArray()));
        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        // Input: [[1,3],[2,6],[8,10],[15,18]]
        // Output: [[1,6],[8,10],[15,18]]
        ArrayList<Interval> intervals = new ArrayList<Interval>( Arrays.asList(new Interval(1,3),
                                                                               new Interval(2,6),
                                                                               new Interval(8,10),
                                                                               new Interval(15,18) ) );
        runTestCase(intervals);

        // Test 2
        // Input: [[1,4],[4,5]]
        // Output: [[1,5]]
        ArrayList<Interval> intervals2 = new ArrayList<Interval>( Arrays.asList(new Interval(1,4),
                                                                                new Interval(4,5) ) );
        runTestCase(intervals2);
    }
}