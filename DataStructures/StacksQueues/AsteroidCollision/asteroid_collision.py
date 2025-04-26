'''
Leetcode 735: Asteroid Collision

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

Example 1:
Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.

Example 2:
Input: asteroids = [8,-8]
Output: []
Explanation: The 8 and -8 collide exploding each other.
'''

from collections import deque
from typing import List

class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        # Stack: keep track of asteroids (parentheses closing)
        stack = []  # deque([])

        # Iterate over each asteroid (left to right by default)
        for asteroid in asteroids:
            # Right: If direction is right, push to stack (cannot collide with previous)
            if asteroid > 0:
                stack.append(asteroid)
            # Left: Direction is left, will collide with previous if there exists one
            else:
                # Destroy previous positive ones
                # While there is at least one asteroid in the stack moving to the right
                # and the current left-moving asteroid is larger (in absolute value) 
                # than the top asteroid in the stack, pop stack top
                while (stack and                  # Stack is not empty
                stack[-1] > 0 and            # Stack top is positive (moving right)
                stack[-1] < -asteroid):   # Absolute value of stack top is smaller than new asteroid
                    stack.pop()               # Pop stock top

                # If stack top has the same size as the current one (but moving in the
                # opposite direction), they both explode.
                if (stack and             # Stack is not empty
                stack[-1] == -asteroid):  # Same absolute value
                    stack.pop()

                # If stack is empty, or stack top is moving left, push current asteroid
                elif not stack or stack[-1] < 0:
                    stack.append(asteroid)

        return stack
