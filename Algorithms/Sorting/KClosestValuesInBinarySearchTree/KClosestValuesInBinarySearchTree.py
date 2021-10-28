"""
K closest values in Binary Search Tree
"""
from typing import List
from heapq import *
import random
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    # Recursive Inorder + Sort: O(n*log(n))
    def closestKValuesRecursiveInorderSort(self, root: TreeNode, target: float, k: int) -> List[int]:
        def inorder(r: TreeNode):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []

        nums = inorder(root)
        nums.sort(key = lambda x: abs(x-target))
        return nums[:k]

    # Recursive Inorder + Heap: O(n*log(k))
    def closestKValues(self, root: TreeNode, target: float, k: int) -> List[int]:
        def inorder(r: TreeNode):
            if not r:
                return
            inorder(r.left)
            # Push to maxHeap (default is minHeap in Python)
            heappush(heap, (-abs(r.val-target), r.val))
            if len(heap) > k:
                heappop(heap)
            inorder(r.right)

        heap = []
        inorder(root)
        return [x for _, x in heap]

    # Quickselect: O(n)
    def closestKValuesQuickselect(self, root: TreeNode, target: float, k: int) -> List[int]:
        def inorder(r: TreeNode):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        def partition(pivot_idx, left, right):
            # 0. Calc pivot distance
            pivot_dist = dist(pivot_idx)
            # 1. Move pivot to end
            nums[right], nums[pivot_idx] = nums[pivot_idx], nums[right]
            store_idx = left
            # 2. move closer elements to left
            for i in range(left, right):
                if dist(i) < pivot_dist:
                    nums[i], nums[store_idx] = nums[store_idx], nums[i]
                    store_idx += 1
            # 3. Move pivot to its final place
            nums[right], nums[store_idx] = nums[store_idx], nums[right]
        def quickselect(left, right):
            if left == right:
                return
            # Select random pivot
            pivot_idx = random.randint(left, right)

            # Find true pivot using partition
            true_idx = partition(pivot_idx, left, right)

            # If final position, return
            if true_idx == k:
                return
            elif true_idx < k:
                # If smaller, go right
                quickselect(true_idx, right)
            # If larger, go left
            else:
                quickselect(left, true_idx)

        nums = inorder(root)
        dist = lambda idx: abs(nums[idx] - target)
        quickselect(0, len(nums)-1)
        return nums[:k]