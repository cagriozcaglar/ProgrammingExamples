/**
 Given a square chessboard of size 8x8, the position of Knight and position of a target is given.
 We need to find out the number of minimum steps a Knight will take to reach the target position.

 Problem in GeeksForGeeks: https://www.geeksforgeeks.org/minimum-steps-reach-target-knight/
 Example solution: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/Shortest%20Knight%20Move.java
 Example solution: http://qa.geeksforgeeks.org/4066/knight-movement-on-the-chess-board
 */

import java.util.*; // For Queue and LinkedList

public class ShortestKnightMove {
    /**
     * Solution: Use BFS to check closest data points. At each step of BFS, distance is increased by 1.
     * This guarantees that the destination will be found with shortest amount of moves.
     *
     * Main solution: https://github.com/tongzhang1994/Facebook-Interview-Coding/blob/master/Shortest%20Knight%20Move.java
     * Example solution: https://www.geeksforgeeks.org/minimum-steps-reach-target-knight/
     *
     *
     * Time complexity: O(n^2), where n is the the size of the square board of size n x n
     *
     * Note: Alternative solution is to use Dijkstra's Shortest Path algorithm.
     * Refer to this solution: https://www.geeksforgeeks.org/greedy-algorithms-set-6-dijkstras-shortest-path-algorithm/
     *
     * @param x1: x-axis of starting point
     * @param y1: y-axis of starting point
     * @param x2: x-axis of destination point
     * @param y2: y-axis of destination point
     * @return
     */
    public static int shortestKnightMove(int x1, int y1, int x2, int y2) {
        // Initialize possible directions a knight can move to: There are 8 different directions a knight can move to
        int[][] directions = new int[][] {
            { 1,  2},
            { 1, -2},
            {-1,  2},
            {-1, -2},
            { 2,  1},
            { 2, -1},
            {-2,  1},
            {-2, -1} };

        // Start BFS
        // Initialize the queue and visiting status array
        Queue<int[]> bfsQueue = new LinkedList<>();
        boolean[][] visited = new boolean[8][8]; // All set to false by default

        // Push the starting point to queue, set visited to true
        int[] source = new int[]{x1, y1}; // 1-D Array of size 2, where the values refer to x and y axis
        bfsQueue.add(source);
        visited[x1][y1] = true;

        // Initialize distance to 0
        int distance = 0;

        // Iterative unitl BFS queue is empty
        while(!bfsQueue.isEmpty()) {
            // Length of BFS queue for the current level. Get it now for the level , because BFS queue size will change in the loop later.
            int sizeForCurrentLevel = bfsQueue.size();

            // Iterate over all nodes in the BFS queue for the current level
            for(int i = 0; i < sizeForCurrentLevel; i++) {
                // Get current position
                int[] position = bfsQueue.poll();
                // Move the knight in each possible direction
                for(int[] direction : directions) {
                    // Update x and y coordinates using current position and direction of the knight move
                    int x = position[0] + direction[0];
                    int y = position[1] + direction[1];

                    // 1. If (x,y) is the target location return the distance
                    if(x == x2 && y == y2) {
                        return distance + 1; // +1, because we make one knight move in this iteration
                    }

                    // 2. If (x,y) is out of bounds or already visited, continue (that is, skip this direction)
                    if( !isWithinBounds(x,y) || visited[x][y] ) {
                        continue;
                    }

                    // 3. If two conditions above are not met, then add this position to BFS queue, to explore later
                    int[] newPoint = new int[]{x, y};
                    bfsQueue.add(newPoint);
                    visited[x][y] = true;
                } // end of iteration over directions of knight moves

            } // end of iterations over all positions in the current level

            // At this point, all positions in the BFS queue for the level are visited. We move to next level, and increment distance.
            distance++;
        } // end of BFS queue iteration

        // At this point, BFS queue is empty, and BFS is complete, and target position is not found. Hence, return -1 as indicator
        return -1;
    }

    /**
     * Helper function
     * @param x
     * @param y
     * @return
     */
    public static boolean isWithinBounds(int x, int y) {
        return (x >= 0) &&
               (x <  8) &&
               (y >= 0) &&
               (y <  8);
    }

    public static void runTestCase(int x1, int y1, int x2, int y2) {
        System.out.println("Minimum number of knight moves from (" + x1 + ", " + y1 + ")" +
                                                           " to (" + x2 + ", " + y2 + ") is: " +
                                                           shortestKnightMove(x1,y1,x2,y2));
    }

    public static void main(String[] args) {
        // Test 1: (3,4) to (0,0): (3,4) -> (4,2) -> (2,1) -> (0,0). 3 moves.
        int x1 = 3;
        int y1 = 4;
        int x2 = 0;
        int y2 = 0;
        runTestCase(x1, y1, x2, y2);

        // Test 2: (7,7) to (0,0): 6 moves.
        x1 = 7;
        y1 = 7;
        x2 = 0;
        y2 = 0;
        runTestCase(x1, y1, x2, y2);
    }
}