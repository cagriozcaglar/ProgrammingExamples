/**
 Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine
 if a person could attend all meetings. For example, Given [[0, 30],[5, 10],[15, 20]], return false.
 */

// Example solution: https://www.programcreek.com/2014/07/leetcode-meeting-rooms-java/

import java.util.*;

public class MeetingRooms {

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
     *
     * @param intervals
     * @return
     */
    public static boolean canAttendAllMeetings(Interval[] intervals) {
        // Boundary cases: return true if null, or if the number of meetings is at most 1
        if(intervals == null || intervals.length <= 1) {
            return true;
        }

        // Sort intervals by start time
        Arrays.sort(intervals, new Comparator<Interval>(){
           public int compare(Interval i1, Interval i2) {
               return i1.start - i2.start;
           }
        });

        // Linear scan of intervals: Check if for all i \in {0,..n-2}, start_{i+1} > end_{i}
        for(int i = 0; i < intervals.length-1; i++) {
            if(intervals[i+1].start < intervals[i].end) {
                return false;
            }
        }

        // If no meeting conflict anywhere in the previous for loop, then attending all meetings is possible
        return true;
    }

    public static void runTestCase(Interval[] intervals) {
        System.out.println("Is it possible to attend all of these meetings with intervals "
                            + Arrays.toString(intervals) + ": "
                            + canAttendAllMeetings(intervals));
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

    }
}