'''
CTCI 6th edition: 17.8: Circus Tower

Circus Tower: A circus is designing a tower routine consisting of people standing atop one another's
shoulders. For practical and aesthetic reasons, each person must be both shorter and lighter than
the person below him or her. Given the heights and weights of each person in the circus, write a
method to compute the largest possible number of people in such a tower.
'''
from typing import List

class Solution:
    def circus_tower(people: List[List[int]]) -> int:
        """
        Find the largest possible tower where each person is both shorter and lighter
        than the person below them.
        
        Args:
            people: A list of (height, weight) tuples representing each person
            
        Returns:
            The maximum number of people possible in the tower and the tower itself
        """
        people.sort(key=lambda x: x[0])  # Sort by height

        n = len(people)
        dp = [1] * n  # Initialize DP array with 1s

        # To reconstruct the tower
        prev = [-1] * n

        for i in range(1, n):
            for j in range(i):
                if people[i][0] < people[j][0] and people[i][1] < people[j][1]:
                    if dp[j] + 1 > dp[i]:
                        dp[i] = dp[j] + 1
                        prev[i] = j

        # Find the index of the maximum value in dp
        max_len = max(dp)
        max_idx = dp.index(max_len)
        
        # Reconstruct the tower
        tower = []
        while max_idx != -1:
            tower.append(people[max_idx])
            max_idx = prev[max_idx]
        
        # Reverse the tower so that the tallest and heaviest person is at the bottom
        tower.reverse()
        
        return max_len, tower
 

def main():
    # Example data: [(height, weight), ...]
    people = [(65, 100), (70, 150), (56, 90), (75, 190), (60, 95), (68, 110)]
    max_tower_size, tower = Solution.circus_tower(people)
    
    print(f"Maximum number of people in the tower: {max_tower_size}")
    print("Tower from bottom to top:")
    for i, (height, weight) in enumerate(tower):
        print(f"Position {i+1}: Height = {height}, Weight = {weight}")
    
if __name__ == "__main__":
    main()

'''
Maximum number of people in the tower: 1
Tower from bottom to top:
Position 1: Height = 56, Weight = 90
'''