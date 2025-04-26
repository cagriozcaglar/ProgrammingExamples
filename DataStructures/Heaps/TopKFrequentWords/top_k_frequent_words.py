'''
Leetcode 692: Top K Frequent Words

Given an array of strings words and an integer k, return the k most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.

Example 1:
Input: words = ["i","love","leetcode","i","love","coding"], k = 2
Output: ["i","love"]
Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.

Example 2:
Input: words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4
Output: ["the","is","sunny","day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words, with the number of occurrence being 4, 3, 2 and 1 respectively.
'''
class Pair:
    def __init__(self, word: str, freq: int):
        self.word = word
        self.freq = freq

    def __lt__(self, other):
        return self.freq < other.freq or \
               (self.freq == other.freq and self.word > other.word)

from typing import List
import heapq
from collections import Counter

class Solution:
    '''
    Solution 1: Max heap
    If we put all numbers into a max heap, the top element of the heap must be the max value of all numbers in the heap.
    So instead of sorting all unique words, we only need to pop the word with the max frequency from a max heap k times.
    - Time Complexity: O(N+klogN). We count the frequency of each word in O(N) time and then heapify the list of unique words in O(N) time. Each time we pop the top from the heap, it costs logN time as the size of the heap is O(N).
    - Space Complexity: O(N), the space used to store our counter cnt and heap h.
    '''
    def topKFrequentMaxHeap(self, words: List[str], k: int) -> List[str]:
        word_freqs = Counter(words)
        word_heap = [(-freq, word) for word, freq in word_freqs.items()]
        heapq.heapify(word_heap)
        return [heapq.heappop(word_heap)[1] for _ in range(k)]

    '''
    Solution 2: Min heap
    Solution 1 looks perfect when the given input is offline, i.e., no new unique words will be added later.
    For those top-k elements problems that may have dynamically added elements, we often solve them by maintaining
    a min heap of size k to store the largest k elements so far. Every time we enumerate a new element, just compare
    it with the top of the min heap and check if it is one of the largest k elements.
    - Time Complexity: O(Nlogk), where N is the length of words. We count the frequency of each word in O(N) time,
    then we add N words to the heap, each in O(logk) time. Finally, we pop from the heap up to k times or just sort
    all elements in the heap as the returned result, which takes O(klogk). As k≤N, O(N) + O(Nlogk) + O(klogk) = O(Nlogk).
    - Space Complexity: O(N), O(N) space is used to store our counter cnt while O(k) space is for the heap.
    '''
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        word_freqs = Counter(words)
        word_heap = []
        for word, freq in word_freqs.items():
            heapq.heappush(word_heap, Pair(word, freq))
            if len(word_heap) > k:
                heapq.heappop(word_heap)
        return [p.word for p in sorted(word_heap, reverse=True)]