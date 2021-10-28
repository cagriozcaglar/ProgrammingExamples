# Solution here: https://github.com/jz33/LeetCodeSolutions/blob/master/T-1188%20Design%20Bounded%20Blocking%20Queue.py
from collections import deque
from threading import Condition

class BoundedBlockingQueue(object):

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = deque()
        self.condition = Condition()

    def enqueue(self, element: int) -> None:
        with self.condition:  # acquire and release
            while len(self.queue) >= self.capacity:
                self.condition.wait()
            self.queue.append(element)
            self.condition.notify()

    def dequeue(self) -> int:
        with self.condition:
            while len(self.queue) == 0:
                self.condition.wait()
            element = self.queue.popleft()
            self.condition.notify()
            return element

    def size(self) -> int:
        with self.condition:
            return len(self.queue)