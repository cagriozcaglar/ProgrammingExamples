'''
Leetcode 855: Exam Room

There is an exam room with n seats in a single row labeled from 0 to n - 1.

When a student enters the room, they must sit in the seat that maximizes the distance to the closest person. If there are multiple such seats, they sit in the seat with the lowest number. If no one is in the room, then the student sits at seat number 0.

Design a class that simulates the mentioned exam room.

Implement the ExamRoom class:
1. ExamRoom(int n) Initializes the object of the exam room with the number of the seats n.
2. int seat() Returns the label of the seat at which the next student will set.
3. void leave(int p) Indicates that the student sitting at seat p will leave the room. It is guaranteed that there will be a student sitting at seat p.
'''

from sortedcontainers import SortedList

class ExamRoom:
    def __init__(self, n: int):
        """Initialize an ExamRoom with n seats."""
        # Helper function to calculate distance between two seats
        def get_distance(pair):
            left, right = pair
            # If seat is at the beginning or end of the row, distance is the number of empty seats
            # Otherwise, distance is the midpoint between left and right seats
            return right - left - 1 if left == -1 or right == n else (right - left) // 2

        self.n = n  # Total number of seats
        # Initialize sorted list to maintain seats and their distances for efficient access
        # Ordered by distance first, then by left seat index
        self.sorted_seats = SortedList(key=lambda x: (-get_distance(x), x[0]))
        self.left_mapping = {}  # Map from seat to the seat to its left
        self.right_mapping = {}  # Map from seat to the seat to its right
        self.add((-1, n))  # Add placeholder for the initial distance from -1 to n

    def seat(self) -> int:
        """Seat a student in the exam room and return the seat number."""
        # Find the seat with the maximum distance to its neighbors
        max_distance_seat = self.sorted_seats[0]
        # Determine the best seat to sit in
        if max_distance_seat[0] == -1:  # Sit at the first seat if it's the best option
            seat_index = 0
        elif max_distance_seat[1] == self.n:  # Sit at the last seat if it's the best option
            seat_index = self.n - 1
        else:  # Otherwise, sit in the middle of the two seats
            seat_index = (max_distance_seat[0] + max_distance_seat[1]) // 2
      
        # Update seat mappings by removing the old pair and adding the new pairs
        self.delete(max_distance_seat)
        self.add((max_distance_seat[0], seat_index))
        self.add((seat_index, max_distance_seat[1]))
      
        # Return the chosen seat index
        return seat_index

    def leave(self, p: int) -> None:
        """A student leaves the seat at index p."""
        # Retrieve neighboring seats
        left_neighbor, right_neighbor = self.left_mapping[p], self.right_mapping[p]
        # Remove the seats that are no longer relevant since the student has left
        self.delete((left_neighbor, p))
        self.delete((p, right_neighbor))
        # Add the new pair created by the student leaving
        self.add((left_neighbor, right_neighbor))

    def add(self, pair):
        """Add a new pair of seats and update their mappings."""
        # Add the pair to the sorted list and update left and right mappings
        self.sorted_seats.add(pair)
        self.left_mapping[pair[1]] = pair[0]
        self.right_mapping[pair[0]] = pair[1]

    def delete(self, pair):
        """Remove a pair of seats and update their mappings."""
        # Remove the pair from the sorted list and delete the mappings
        self.sorted_seats.remove(pair)
        self.left_mapping.pop(pair[1], None)
        self.right_mapping.pop(pair[0], None)