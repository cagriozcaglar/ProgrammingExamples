'''
Leetcode 981: Time Based Key-Value Store

Design a time-based key-value data structure that can store multiple values for the same key at different time stamps
and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:

1) TimeMap() Initializes the object of the data structure.
2) void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
3) String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp.
If there are multiple such values, it returns the value associated with the largest timestamp_prev.
If there are no values, it returns "".
'''

from typing import Dict, List, Tuple
from collections import defaultdict
from sortedcontainers import SortedDict

# Using SortedDict
class TimeMap:

    def __init__(self):
        # HashMap[ key: str, SortedDict[timestamp: int, value: str] ]
        # Inside map is a SortedDict, because we want to search for timestamps using Binary Search
        self.key_time_map: Dict[str, SortedDict[int, str]] = defaultdict(SortedDict)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # Because of the use of defaultdict, we don't need to check if key exists
        self.key_time_map[key][timestamp] = value

    def get(self, key: str, timestamp: int) -> str:
        # Find element on sorted timestamps using Binary Search
        if key not in self.key_time_map:
            return ""

        # Find first value greater than timestamp.
        iterator = self.key_time_map[key].bisect_right(timestamp)

        # A Timestamp before timestamp doesn't exist, returning iterator == 0
        if iterator == 0:
            return ""

        # Return the value stored in previous position of current iterator
        # which returns the greatest timestamp smaller than timestamp variable
        return self.key_time_map[key].peekitem(iterator - 1)[1]

# Using Binary Search
class TimeMap2:

    def __init__(self):
        self.key_time_map: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        
    def set(self, key: str, value: str, timestamp: int) -> None:
        self.key_time_map[key].append((timestamp, value))

    # Get using Binary search
    def get(self, key: str, timestamp: int) -> str:
        if key not in self.key_time_map:
            return ""
                                
        if timestamp < self.key_time_map[key][0][0]:
            return ""
                                
        left, right = 0, len(self.key_time_map[key])
                                
        while left < right:
            mid = (left + right) // 2
            if self.key_time_map[key][mid][0] <= timestamp:
                left = mid + 1
            else:
                right = mid
                            
        return "" if right == 0 else self.key_time_map[key][right-1][1]