/**
 Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] find the minimum
 number of conference rooms required.
 */

// Example solution: https://www.programcreek.com/2014/05/leetcode-meeting-rooms-ii-java/
// Example solution: http://happycoding2010.blogspot.com/2015/11/leetcode-253-meeting-rooms-ii.html
// Example solution: http://fisherlei.blogspot.com/2017/08/leetcode-meeting-rooms-ii-solution.html
import java.util.*;

public class MeetingRoomsII {

    /**
     * Interval class
     */
    static class Interval {
        int start;
        int end;
        Interval(int start, int end) {
            this.start = start;
            this.end = end;
        }

        // Override toString() method for nice printing of Interval objects
        public String toString() {
            return "(" + start + "," + end + ")";
        }
    }

    /**
     * Solution: Using heap, method 1
     * @param intervals
     * @return
     */
    public static int getMinimumNumberOfMeetingRooms1(Interval[] intervals) {
        // Boundary cases: return true if null, or if the number of meetings is at most 1
        if(intervals == null) {
            return 0;
        }

        // Sort intervals by start time
        Arrays.sort(intervals, new Comparator<Interval>(){
            public int compare(Interval i1, Interval i2) {
                return i1.start - i2.start;
            }
        });

        // Create Priority Queue (using Heap) of end times. This is a.k.a the heap of unfinished events.
        PriorityQueue<Integer> endTimeHeap = new PriorityQueue<Integer>();
        for(int i = 0; i < intervals.length; i++) {
            // If not first interval, and start time of current interval is larger than closest end time,
            if( (i > 0) && (intervals[i].start > endTimeHeap.peek()) ) {
                endTimeHeap.poll(); // Close the event
            }
            endTimeHeap.add(intervals[i].end);
        }

        // Return size of endTimeHeap
        return endTimeHeap.size();
    }

    /**
     * Solution: Keeping separate sorted arrays of start times and end times
     * @param intervals
     * @return
     */
    public static int getMinimumNumberOfMeetingRooms2(Interval[] intervals) {
        // Get start / end times into an array and sort
        int[] startTimes = new int[intervals.length];
        int[] endTimes = new int[intervals.length];
        for(int i = 0; i < intervals.length; i++) {
            startTimes[i] = intervals[i].start;
            endTimes[i] = intervals[i].end;
        }
        Arrays.sort(startTimes);
        Arrays.sort(endTimes);

        // Iterate over startTimes and endTimes in parallel, and keep maximum difference between meetings started vs. ended
        int maxDifference = 0;
        for(int startPointer = 0, endPointer = 0; startPointer < intervals.length; ) {
            if(startTimes[startPointer] < endTimes[endPointer]) { // If an event started before next end event, increment startPointer
                startPointer++;
            } else if(startTimes[startPointer] > endTimes[endPointer]) { // If an event ended before next start event, increment endPointer
                endPointer++;
            } else { // If a start event and end event happened at the same time, increment both startPointer and endPointer
                startPointer++;
                endPointer++;
            }

            // Update difference
            maxDifference = Math.max(maxDifference, (startPointer-endPointer));
        }
        return maxDifference;
    }

    public static void runTestCase(Interval[] intervals) {
        // Method 1
        System.out.println("Minimum number of conference rooms using method 1 for these meetings with intervals "
                + Arrays.toString(intervals) + ": "
                + getMinimumNumberOfMeetingRooms1(intervals));

        // Method 2
        System.out.println("Minimum number of conference rooms using method 2 for these meetings with intervals "
                + Arrays.toString(intervals) + ": "
                + getMinimumNumberOfMeetingRooms2(intervals));

        System.out.println();
    }

    public static void main(String[] args) {
        // Test 1
        Interval[] intervals1 = new Interval[]{
                new Interval(1, 3),
                new Interval(4, 7),
                new Interval(20, 90)
        };
        runTestCase(intervals1);

        // Test 2
        Interval[] intervals2 = new Interval[]{
                new Interval(1, 5),
                new Interval(2, 7),
                new Interval(9, 90)
        };
        runTestCase(intervals2);

        // Test 3
        Interval[] intervals3 = new Interval[]{
                new Interval(1, 5),
                new Interval(1, 10),
                new Interval(2, 90),
                new Interval(3, 100),
                new Interval(4, 110)
        };
        runTestCase(intervals3);
    }
}