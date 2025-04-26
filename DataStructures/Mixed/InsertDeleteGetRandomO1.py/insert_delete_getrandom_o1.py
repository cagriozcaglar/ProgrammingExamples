'''
Leetcode 380: Insert Delete GetRandom O(1)

Implement the RandomizedSet class:
- RandomizedSet() Initializes the RandomizedSet object.
- bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
- bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
- int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.

You must implement the functions of the class such that each function works in average O(1) time complexity.
'''

import random
from typing import Dict, List
from collections import defaultdict

class RandomizedSet:

    def __init__(self):
        self.array: List[int] = []
        self.val_index_map: Dict[int, int] = defaultdict(int)
        

    def insert(self, val: int) -> bool:
        if val not in self.val_index_map:
            self.array.append(val)
            self.val_index_map[val] = len(self.array) - 1
            return True
        return False
        

    def remove(self, val: int) -> bool:
        if val in self.val_index_map:
            # Insertions first
            index = self.val_index_map[val]
            self.array[index] = self.array[-1]
            self.val_index_map[self.array[index]] = index
            # Deletions second, from array and hash_map
            self.array.pop()
            del self.val_index_map[val]
            return True
        return False
        

    def getRandom(self) -> int:
        index = random.choice(range(len(self.array)))
        return self.array[index]
                              

# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()