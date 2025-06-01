'''
Leetcode 332: Reconstruct Itinerary

You are given a list of airline tickets where tickets[i] = [fromi, toi] represent the departure and the arrival
airports of one flight. Reconstruct the itinerary in order and return it.

All of the tickets belong to a man who departs from "JFK", thus, the itinerary must begin with "JFK". If there are
multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single
string.

For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].

You may assume all tickets form at least one valid itinerary. You must use all the tickets once and only once.

Example 1:
Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
Output: ["JFK","MUC","LHR","SFO","SJC"]
'''
from typing import List
from collections import defaultdict

class Solution:
    def findItinerary(self, tickets: List[List[str]]) -> List[str]:
        # Create defaultdict representing source -> dest_list
        graph = defaultdict(list)

        # Sort tickets in reverse, because we will pop destinations from the end of the list
        # which is more efficient than pop(0), and we need smallest lexical order for equal paths
        for departure, arrival in sorted(tickets, reverse=True):
            graph[departure].append(arrival)

        # Init a list to keep track of itinerary
        itinerary = []

        # DFS
        def dfs(airport):
            # While there are destinations to visit from current airport
            while graph[airport]:
                # Visit destination by doing DFS on the last airport in the list
                # Since it is reverse-sorted, smallest lexical goes last.
                dfs(graph[airport].pop())
            # Append airport to itinerary after visiting all destinations
            itinerary.append(airport)

        # Begin DFS with JFK
        dfs("JFK")

        # Itinerary is in reverse order due to DFS reverse it
        return itinerary[::-1]