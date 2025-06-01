'''
Leetcode 1244: Design Leaderboard

Design a Leaderboard class, which has 3 functions:

- addScore(playerId, score): Update the leaderboard by adding score to the given player's score. If there is no player with such id in the leaderboard, add him to the leaderboard with the given score.
- top(K): Return the score sum of the top K players.
- reset(playerId): Reset the score of the player with the given id to 0 (in other words erase it from the leaderboard). It is guaranteed that the player was added to the leaderboard before calling this function.

Initially, the leaderboard is empty. 
'''

'''
Time Complexity:
O(1) for addScore.
O(1) for reset.
O(K)+O(NlogK) = O(NlogK). It takes O(K) to construct the initial heap and then for the rest of the N−K elements, we perform the extractMin and add operations on the heap each of which take (logK) time.

Space Complexity:
O(N+K) where O(N) is used by the scores dictionary and O(K) is used by the heap.
'''
import heapq

class Leaderboard:

    def __init__(self):
        self.scores = {}
        

    def addScore(self, playerId: int, score: int) -> None:
        if playerId not in self.scores:
            self.scores[playerId] = 0
        self.scores[playerId] += score
        

    def top(self, K: int) -> int:
        # min-heap
        heap = []
        for x in self.scores.values():
            heapq.heappush(heap, x)
            if len(heap) > K:
                heapq.heappop(heap)
        res = 0
        while heap:
            res += heapq.heappop(heap)
        return res

    def reset(self, playerId: int) -> None:
        self.scores[playerId] = 0
        


# Your Leaderboard object will be instantiated and called as such:
# obj = Leaderboard()
# obj.addScore(playerId,score)
# param_2 = obj.top(K)
# obj.reset(playerId)