'''
Leetcode 729: My Calendar I

You are implementing a program to use as your calendar. We can add a new event if adding the event will not cause a double booking.

A double booking happens when two events have some non-empty intersection (i.e., some moment is common to both events.).

The event can be represented as a pair of integers startTime and endTime that represents a booking on the half-open interval [startTime, endTime), the range of real numbers x such that startTime <= x < endTime.

Implement the MyCalendar class:

MyCalendar() Initializes the calendar object.
boolean book(int startTime, int endTime) Returns true if the event can be added to the calendar successfully without causing a double booking. Otherwise, return false and do not add the event to the calendar.

Example 1:

Input
["MyCalendar", "book", "book", "book"]
[[], [10, 20], [15, 25], [20, 30]]
Output
[null, true, false, true]

Explanation
MyCalendar myCalendar = new MyCalendar();
myCalendar.book(10, 20); // return True
myCalendar.book(15, 25); // return False, It can not be booked because time 15 is already booked by another event.
myCalendar.book(20, 30); // return True, The event can be booked, as the first event takes every time less than 20, but not including 20.
'''
from sortedcontainers import SortedList

'''
Approach 2: Sorted Lists + Binary Search
- Time Complexity: O(NlogN). For each new event, we search that the event is legal in O(logN) time, then insert it in O(logN) time.
- Space Complexity: O(N), the size of the data structures used.
'''
class MyCalendar:
    def __init__(self):
        self.calendar = SortedList()

    def book(self, startTime: int, endTime: int) -> bool:
        idx = self.calendar.bisect_right((startTime, endTime))
        # Cases for conflicting events:
        # Case 1: End of previous > start of current
        # Case 2: Start of current < end
        if (idx > 0 and self.calendar[idx-1][1] > startTime) or \
           (idx < len(self.calendar) and self.calendar[idx][0] < endTime):
           return False
        # IMPORTANT: SortedList has add() method, not append() method.
        self.calendar.add((startTime, endTime))
        return True

'''
Approach #1: Brute Force
- Time Complexity: O(N^2). For each new event, we process every previous event to decide whether the new event can be booked. This leads to ∑_k^N O(k) = O(N^2) complexity.
- Space Complexity: O(N), the size of the calendar.
'''
class MyCalendarBruteForce:
    def __init__(self):
        self.calendar = []

    def book(self, startTime: int, endTime: int) -> bool:
        for s, e in self.calendar:
            if s < endTime and startTime < e:
                return False
        self.calendar.append((startTime, endTime))
        return True


# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(startTime,endTime)