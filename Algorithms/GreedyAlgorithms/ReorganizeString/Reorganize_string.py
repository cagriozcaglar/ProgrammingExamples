'''
Leetcode 767: Reorganize String

Given a string s, rearrange the characters of s so that any two adjacent characters are not the same.

Return any possible rearrangement of s or return "" if not possible.

Example 1:
Input: s = "aab"
Output: "aba"

Example 2:
Input: s = "aaab"
Output: ""
'''
import heapq
from collections import Counter

class Solution:
    '''
    # Solution: Heap + Counting
    # Let N be the total characters in the string.
    # Let k be the total unique characters in the string.
    # Time: O(N * log(k))
    # Time complexity: O(N⋅logk). We add one character to the string per iteration, so there are O(N) iterations. In each iteration, we perform a maximum of 3 priority queue operations. Each priority queue operation costs logk. For this problem, k is bounded by 26, so one could argue that the time complexity is actually O(N).
    # Space complexity: O(k). The counter used to count the number of occurrences will incur a space complexity of O(k). Similarly, the maximum size of the priority queue will also be O(k). Given that k <= 26 in this problem, one could argue the space complexity is O(1).
    '''
    def reorganizeString(self, s: str) -> str:
        # Initialize an empty list ans to store the rearranged characters.
        answer = []
        freqs = dict(Counter(s))

        # max_heap: Use negative values for count, because Python uses min-heap
        pq = [(-count, char) for char, count in freqs.items()]
        heapq.heapify(pq)

        while pq:
            # Most frequent character
            count_first, char_first = heapq.heappop(pq)  # Pop the most frequent character
            if not answer or char_first != answer[-1]:
                answer.append(char_first)
                if count_first + 1 != 0:  # Because counts are negative
                    # Push most frequent character back to heap, with decreased count (increased because of negative values)
                    heapq.heappush(pq, (count_first + 1, char_first))  
            else:
                if not pq:
                    return ''
                # Second most frequent character
                count_second, char_second = heapq.heappop(pq)  # Pop the most frequent character
                answer.append(char_second)
                if count_second + 1 != 0:
                    # Push second most frequent character back to heap, with decreased count (increased because of negative values)
                    heapq.heappush(pq, (count_second + 1, char_second))
                # TODO: Pushing first char again, but why?
                heapq.heappush(pq, (count_first, char_first))

        return ''.join(answer)
