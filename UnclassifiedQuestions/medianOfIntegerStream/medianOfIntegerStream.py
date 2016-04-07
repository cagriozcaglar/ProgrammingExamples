### Given a stream of integers, find the running median element of the stream at any given time.
'''
Solution:
Use two data structures: a max-heap and a min-heap.
-- Max-heap will contain the smallest half of the stream of integers.
   Root of max-heap will be the largest element of the smallest half of integer stream.
-- Min-heap will contain the largest half of the stream of integers.
   Root of min-heap will be the smallest element of the largest half of integer stream.
-- If we received 2N elements (even), max-heap and min-heap will contain N elements each.
   if we received 2N+1 elements (odd), max-heap will contain N+1 elements, min-heap will contain N elements.
-- At time t, if there are k elements in the integer stream:
   -- If k is odd, the median is the root of the max-heap.
   -- If k is even, the median is the average of the root of max-heap and min-heap.
'''
# Example solution: http://www.ardendertat.com/2011/11/03/programming-interview-questions-13-median-of-integer-stream/

import heapq

class medianOfIntegerStream:
    def __init__(self):
        # Number of elements in the stream.
        self.size=0
        # min-Heap holds the largest half. Root is the smallest element of the largest half.
        self.minHeap = []
        # max-Heap holds the smalles half. Root is the largest element of the smallest half.
        self.maxHeap = []
    def insertElement(self, element):
        # Size is even
        if(self.size % 2 ==0):
            # Python has min-heap implementation, therefore multiply element with -1 to main max-heap property.
            heapq.heappush(self.maxHeap, (-1)*element)
            # Increment the size of the integer stream
            self.size += 1
            if(len(self.minHeap)==0):
                return
            if( (-1)*self.maxHeap[0] > self.minHeap[0]):
                # Swap the root elements of min-heap and max-heap
                # Pop root from max-heap and min-heap to be swapped
                elementForMinHeap = (-1) * heapq.heappop(self.maxHeap)
                elementForMaxHeap = heapq.heappop(self.minHeap)
                # Push the root of min-heap to max-heap, and the root of max-heap to min-heap
                heapq.heappush(self.maxHeap, (-1) * elementForMaxHeap)
                heapq.heappush(self.minHeap, elementForMinHeap)
        # Size is odd
        else:
            # Push new element to min-heap to keep the size balance.
            # heapq.heappushpop() method: Push item on the heap, then pop and return the smallest item from the heap. 
            # The combined action runs more efficiently than heappush() followed by a separate call to heappop().
            elementForMinHeap = (-1) * heapq.heappushpop(self.maxHeap, (-1)*element)
            # Now max-heap has N+2 elements, min-heap has N elements, so we need to balance.
            heapq.heappush(self.minHeap, elementForMinHeap)
            # Increment the size of the integer stream
            self.size += 1
    # Get median
    def getMedian(self):
        # If size is even, return the average of the two middle elements, which are at the roots of max-heap and min-heap
        if(self.size % 2 ==0):
            return ( (-1)*self.maxHeap[0] + self.minHeap[0]) / 2.0
        # If size is odd, return the middle element, which is at the root of max-heap
        else:
            return ( (-1)*self.maxHeap[0] )

# Create a stream of integers and calculate the median
intStream = medianOfIntegerStream()
intStream.insertElement(3)
intStream.insertElement(8)
print("Elements are: [3,8]. Median is " + str(intStream.getMedian()) + ".")
intStream.insertElement(5)
print("Elements are: [3,8,5]. Median is " + str(intStream.getMedian()) + ".")
intStream.insertElement(4)
print("Elements are: [3,8,5,4]. Median is " + str(intStream.getMedian()) + ".")
intStream.insertElement(10)
print("Elements are: [3,8,5,4,10]. Median is " + str(intStream.getMedian()) + ".")

''' Output:
Elements are: [3,8]. Median is 5.5
Elements are: [3,8,5]. Median is 5
Elements are: [3,8,5,4]. Median is 4.5
Elements are: [3,8,5,4,10]. Median is 5
'''
