# HashMap: Integer -> Integer
# key: Integer
# hash(key) = 2^key % (size) => [0, size-1]
"""
HashMap:
size
hash(int): int

hash(400) = 1

hash(key)
0 -> [ [] ]
1 -> [ [200, 2], [300, 3], [400, 4] ]
2 ->
3 -> [200]
4
5

# Collisions
# 1. Chaining (*)
# 2. Open addressing

HashMap.get(300) = val => hash(1) ->

hash(300) -> 1

"""
from typing import List

class HashMap:
    def __init__(self, size):
        self.size = size
        self.internalMap = [ [] for _ in range(self.size)] # Assumed size

    # TODO: Add another get method which returns result

    def hash(self, key: int) -> int:
        # Hash function: 2 ^ key % self.size
        # TODO: Check possible overflow
        return (2 ** key) % self.size

    def get(self, key: int) -> List:
        """
        300 -> hash(300) = 1
        1 -> [ [200, 2], [300, 3], [400, 4] ]
        """
        #val = [[200, 2], [300, 3], [400, 4]][0]
        hash: int = self.hash(key)
        listOfValues = self.internalMap[hash]
        print(f"listOfValues: {listOfValues}")
        for keyValuePair in listOfValues:
            keyCheck, value = keyValuePair
            print(f"keyCheck: {keyCheck}")
            print(f"value: {value}")
            if keyCheck == key:
                return [True, value]
        else: # Check
            #except Exception:
            print(f"Key not found: {key}")
            return [False, 0]

    def put(self, key: int, value: int) -> None:
        """
        300 -> hash(300) = 1
        1 -> [ [200, 2], [300, 3], [400, 4] ]
        """
        # Check size constraint
        # hash: int = self.hash(key)
        keyExists, valueCurrent = self.get(key)
        hash = self.hash(key)
        # Case 1: Key exists, update
        if keyExists and valueCurrent != value:
            listOfValues = self.internalMap[hash]
            for keyValuePair in listOfValues:
                keyCheck, valueCheck = keyValuePair
                if keyCheck == key:
                    self.internalMap[hash][keyCheck] = valueCheck
        # Case 2: Key does not exist
        elif not keyExists:
            self.internalMap[hash].append( [key, value] )


def main():
    size = 3
    myMap: HashMap = HashMap(size)
    myMap.put(1,3)
    print(f"myMap.internalMap: {myMap.internalMap}")
    myMap.put(10,30)
    print(f"myMap.internalMap: {myMap.internalMap}")
    myMap.put(102,300)
    print(f"myMap.internalMap: {myMap.internalMap}")
    myMap.get(1)
    print(f"myMap.get(1): {myMap.get(1)[-1]}")
    myMap.get(80)
    print(f"myMap.get(80): {myMap.get(80)}")

    """
    for key in myMap.internalMap:
        print(f"key: {key}")
        for key1, value1 in myMap.internalMap[key]:
            print(f"key:{key1}, value:{value1}")
    """
main()