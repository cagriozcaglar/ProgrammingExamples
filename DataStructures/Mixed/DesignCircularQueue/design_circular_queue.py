'''
Leetcode 622: Design Circular Queue

Design your implementation of the circular queue. The circular queue is a linear data structure in which the operations are performed based on FIFO (First In First Out) principle, and the last position is connected back to the first position to make a circle. It is also called "Ring Buffer".

One of the benefits of the circular queue is that we can make use of the spaces in front of the queue. In a normal queue, once the queue becomes full, we cannot insert the next element even if there is a space in front of the queue. But using the circular queue, we can use the space to store new values.

Implement the MyCircularQueue class:

MyCircularQueue(k) Initializes the object with the size of the queue to be k.
int Front() Gets the front item from the queue. If the queue is empty, return -1.
int Rear() Gets the last item from the queue. If the queue is empty, return -1.
boolean enQueue(int value) Inserts an element into the circular queue. Return true if the operation is successful.
boolean deQueue() Deletes an element from the circular queue. Return true if the operation is successful.
boolean isEmpty() Checks whether the circular queue is empty or not.
boolean isFull() Checks whether the circular queue is full or not.
You must solve the problem without using the built-in queue data structure in your programming language. 
'''

'''
Using array (self.queue), head_index, count, capacity
- Time complexity: O(1). All of the methods in our circular data structure is of constant time complexity.
- Space Complexity: O(N). The overall space complexity of the data structure is linear, where N is the pre-assigned capacity of the queue. However, it is worth mentioning that the memory consumption of the data structure remains as its pre-assigned capacity during its entire life cycle.
'''
from threading import Lock

class MyCircularQueue:
    def __init__(self, k: int):
        self.queue = [0] * k
        self.head_index = 0
        self.count = 0
        self.capacity = k
        self.queue_lock = Lock()

    def enQueue(self, value: int) -> bool:
        with self.queue_lock:
            if self.count == self.capacity:
                return False
            self.queue[(self.head_index + self.count) % self.capacity] = value
            self.count += 1
        # Release lock when leaving the block
        return True

    def deQueue(self) -> bool:
        if self.count == 0:
            return False
        self.head_index = (self.head_index + 1) % self.capacity
        self.count -= 1
        return True

    def Front(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[self.head_index]

    def Rear(self) -> int:
        if self.count == 0:
            return -1
        return self.queue[(self.head_index + self.count - 1) % self.capacity]

    def isEmpty(self) -> bool:
        return self.count == 0

    def isFull(self) -> bool:
        return self.count == self.capacity