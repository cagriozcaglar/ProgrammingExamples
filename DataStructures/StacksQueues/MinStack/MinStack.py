"""
Leetcode 155: Min Stack 

Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

push(x) -- Push element x onto stack.
pop() -- Removes the element on top of the stack.
top() -- Get the top element.
getMin() -- Retrieve the minimum element in the stack.

Example 1:
- Input
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]]
- Output
[null,null,null,null,-3,null,0,-2]
Explanation
MinStack minStack = new MinStack();
minStack.push(-2);
minStack.push(0);
minStack.push(-3);
minStack.getMin(); // return -3
minStack.pop();
minStack.top();    // return 0
minStack.getMin(); // return -2

Constraints:
- Methods pop, top and getMin operations will always be called on non-empty stacks.
"""
from collections import deque

class MinStack:
    def __init__(self):
        self.stack = []
        
    def push(self, val: int) -> None:
        if not self.stack:
            self.stack.append((val, val))
            return
        current_min = self.stack[-1][1]
        self.stack.append((val, min(val, current_min)))
        
    def pop(self) -> None:
        self.stack.pop()
        
    def top(self) -> int:
        return self.stack[-1][0]
        
    def getMin(self) -> int:
        return self.stack[-1][1]


class MinStack2:
    def __init__(self):
        """
        initialize MinStack with two stacks
         - originalStack: Holds original stack values.
         - minTrackerStack: Tracks minimum value, updated based on pop / push ops, maintains min-so-far at top
        """
        self.originalStack = deque()
        self.minTrackerStack = deque()

    def push(self, x: int) -> None:
        # Update minTrackerStack
        # 1.1. Empty stacks => Always push to minTrackerStack
        # 1.2. If x is <= min value (at minTrackerStack[-1]) so far, push new min value to minTrackerStack.
        # Note: Even in equality case, push to minTrackerStack
        if (len(self.originalStack) == 0 and len(self.minTrackerStack) == 0) or \
                (len(self.minTrackerStack) > 0 and x <= self.minTrackerStack[-1]):
            self.minTrackerStack.append(x)
        # Update originalStack
        # 2. Regardless, always push x to originalStack
        self.originalStack.append(x)

    def pop(self) -> None:
        # If x == min value (at minTrackerStack[-1]) so far, pop min value from minTrackerStack
        if len(self.originalStack) > 0:
            # Note: if self.originalStack is non-empty, self.minTrackerStack is non-empty as well
            if self.originalStack[-1] == self.minTrackerStack[-1]:
                self.minTrackerStack.pop()
            # Regardless, always pop from originalStack
            self.originalStack.pop()
        # If self.originalStack is empty, skip, don't do anything

    def top(self) -> int:
        # Return top of originalStack
        # NOTE: Whenever you need to pop or top a stack, check if it is empty
        if len(self.originalStack) > 0:
            return self.originalStack[-1]
        else:
            return -1

    def getMin(self) -> int:
        # Return top of minTrackerStack
        if len(self.minTrackerStack) > 0:
            return self.minTrackerStack[-1]
        else:
            return -1


def printMinStackContent(minStack: MinStack):
    print(f"minStack.originalStack: {str(minStack.originalStack)}, "
          f"minStack.minTrackerStack: {str(minStack.minTrackerStack)}")


if __name__ == "__main__":
    # Example 1:
    minStack: MinStack = MinStack()
    printMinStackContent(minStack)
    minStack.push(-2)  # OriginalStack: [-2], MinTrackerStack: [-2]
    printMinStackContent(minStack)
    minStack.push(0)   # OriginalStack: [-2, 0], MinTrackerStack: [-2]
    printMinStackContent(minStack)
    minStack.push(-3)  # OriginalStack: [-2, 0, -3], MinTrackerStack: [-2, -3]
    printMinStackContent(minStack)
    print(f"minStack.getMin(): {minStack.getMin()}")  # return -3
    minStack.pop()     # OriginalStack: [-2, 0], MinTrackerStack: [-2]
    printMinStackContent(minStack)
    print(f"minStack.top(): {minStack.top()}")  # return -3
    print(f"minStack.getMin(): {minStack.getMin()}")  # return -2
