'''
CTCI 8.13: Stack of Boxes

You have a stack of n boxes, with widths W i' heights hi' and depths d1• The boxes
cannot be rotated and can only be stacked on top of one another if each box in the stack is strictly
larger than the box above it in width, height. and depth. Implement a method to compute the
height of the tallest possible stack. The height of a stack is the sum of the heights of each box.
'''

'''
Solution 1:
    We basically turn this into a LIS (longest increasing subsequence) problem. Sort the boxes by height in decreasing order.
    Then make a DP arr and initalize each index to the height of each respective box. Then for each box,
    check if any previous box was bigger by all dimensions and if there is such a box, set the current box's dp val (max height)
    to max(current_box max_height, found box's max_height + current_box height). Finally, since the dp arr represents
    the max height of a stacks with the box at an index being the top box in the given stack, we just find the max value in the 
    dp arr to get the height of the tallest stack possible.
'''
import unittest
from functools import reduce


def stack_of_boxes(boxes):
    # width: 0, height: 1, depth: 2. Sorting by height in decreasing order.
    boxes = sorted(boxes, key=lambda x: x[1], reverse=True)
    # Initialize dp array with height[i] for each box at index i
    dp = [boxes[i][1] for i in range(len(boxes))]
    
    # For each index i, walk through all indices j before i
    for i in range(1, len(boxes)):
        for j in range(i):
            # if all dimensions are higher for j
            if boxes[j][0] > boxes[i][0] and boxes[j][1] > boxes[i][1] and boxes[j][2] > boxes[i][2]:
                dp[i] = max(dp[i], dp[j] + boxes[i][1])
    
    return max(dp)

'''
Solution 2:
Use objects instead. Also, look forward, instead of backward
'''
class Box:
    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth
    
    def __lt__(self, other):
        return self.height < other.height
    
    def __eq__(self, other):
        return self.height == other.height

def stack_of_boxes_2(boxes):
    boxes.sort(reverse=True)  # Sort by height in decreasing order, in-place

    def tallest_for_bottom(cur_stack, cur_box_idx):
        if cur_box_idx == len(boxes):
            return reduce(lambda x, y: x + y.height, cur_stack, 0)
        
        if cur_stack[-1].height > boxes[cur_box_idx].height and \
           cur_stack[-1].width > boxes[cur_box_idx].width and \
           cur_stack[-1].depth > boxes[cur_box_idx].depth:
            return tallest_for_bottom(cur_stack + [boxes[cur_box_idx]], cur_box_idx+1)
        
        return tallest_for_bottom(cur_stack, cur_box_idx+1)

    largest_height = 0
    for i, box in enumerate(boxes):
        largest_height = max(largest_height, tallest_for_bottom([box], i+1))

    return largest_height

def test_null():
    assert stack_of_boxes_2([]) == 0


def test_single_box():
    assert stack_of_boxes_2([Box(3, 2, 1)]) == 3


def test_two_conflicting_boxes():
    assert stack_of_boxes_2([Box(3, 2, 1), Box(5, 4, 1)]) == 5


def test_two_stackable_boxes():
    assert stack_of_boxes_2([Box(3, 2, 1), Box(6, 5, 4)]) == 9


if __name__ == "__main__":
    unittest.main()