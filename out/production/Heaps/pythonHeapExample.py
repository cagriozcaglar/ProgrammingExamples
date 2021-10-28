import heapq

print("Python Heap Example starts")
# Initialize a heap (always a min-heap in Python) as a list
theHeap = []

# Push elements to min-heap
heapq.heappush(theHeap, 1)
heapq.heappush(theHeap, 10)
heapq.heappush(theHeap, 5)
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
1
10
5
'''

# Remove the element at the root of the (min-)heap (minimum element)
heapq.heappop(theHeap)
'''
1
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
5
10
'''

# Push 9 on the heap, pop and return the smallest element of the heap
heapq.heappushpop(theHeap,9)
'''
5
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
9
10
'''

# Pop and return the smallest element from the heap, and then push the new item
heapq.heapreplace(theHeap,11)
'''
9
'''
# Print the contents of the heap, written from root to down
for element in theHeap:
    print(element)
'''
10
11
'''

# Create a heap from a list in-place, in-linear time
newHeap = [9,8,7,6,5,4,3,2,1]
heapq.heapify(newHeap)
# Print the contents of the heap, written from root to down
for element in newHeap:
    print(element)
'''
1
2
3
6
5
4
7
8
9
'''

# Get 4 smallest elements from the heap
smallestElements = heapq.nsmallest(4,newHeap)
print(smallestElements)
'''
[1, 2, 3, 4]

'''

# get 3 largest elements from the heap
largestElements = heapq.nlargest(3,newHeap)
print(largestElements)
'''
[9, 8, 7]
'''
print("Python Heap Example ends")
