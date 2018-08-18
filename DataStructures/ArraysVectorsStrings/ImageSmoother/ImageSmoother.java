/**
 Given a 2D integer matrix M representing the gray scale of an image, you need to design a smoother to make the gray scale
 of each cell becomes the average gray scale (rounding down) of all the 8 surrounding cells and itself. If a cell has less
 than 8 surrounding cells, then use as many as you can.

 Example 1:
 Input:
 [[1,1,1],
 [1,0,1],
 [1,1,1]]
 Output:
 [[0, 0, 0],
 [0, 0, 0],
 [0, 0, 0]]
 Explanation:
 For the point (0,0), (0,2), (2,0), (2,2): floor(3/4) = floor(0.75) = 0
 For the point (0,1), (1,0), (1,2), (2,1): floor(5/6) = floor(0.83333333) = 0
 For the point (1,1): floor(8/9) = floor(0.88888889) = 0

 Note:
 The value in the given matrix is in the range of [0, 255].
 The length and width of the given matrix are in the range of [1, 150].
 */

class ImageSmoother {
    /**
     * Image smoother 1: Checks bounds with non-modularized / inline condition checks
     * @param M
     * @return
     */
    public static int[][] imageSmoother1(int[][] M) {
        if (M == null) {
            return null;
        }
        int[][] smoother = new int[M.length][M[0].length];
        // Go through the matrix values and take average values
        for(int row = 0; row < M.length; row++){
            for(int column = 0; column < M[0].length; column++){
                int sum = 0;
                int count = 0;
                // Check the neigbours. If the neighbour is within bounds, add to the sum and increase count
                for(int r = row-1; r <= row+1 ; r++) {   // r in [row-1, row+1]
                    for(int c = column-1; c <= column+1; c++) {   // c in [column-1, column+1]
                        if((r >= 0 && r < M.length) && (c >= 0 && c < M[0].length)){  // Check if within bounds
                            sum += M[r][c];
                            count++;
                        }
                    }
                }
                // Note: Check for zero counts, to prevent divide by zero errors.
                smoother[row][column] = (count > 0) ? (sum / count) : 0;
                //smoother[row][column] = sum / count;
            }
        }
        return smoother;
    }

    /**
     * Image smoother 2: Checks bounds with modularized condition checks. Cleaner version.
     * Helper function isWithinBounds(int r, int c, int numRows, int numColumns) is used.
     * @param M
     * @return
     */
    public static int[][] imageSmoother2(int[][] M) {
        if (M == null) {
            return null;
        }
        int[][] smoother = new int[M.length][M[0].length];
        // Go through the matrix values and take average values
        for(int row = 0; row < M.length; row++){
            for(int column = 0; column < M[0].length; column++){
                int sum = 0;
                int count = 0;
                // Check the neigbours using incremental values in int[]{-1,0,1} (nice trick).
                // If the neighbour is within bounds, add to the sum and increase count
                for(int incRow : new int[]{-1,0,1}) {
                    for(int incColumn : new int[]{-1,0,1}) {
                        if( isWithinBounds(row+incRow, column+incColumn, M.length, M[0].length) ){ // Check if within bounds
                            sum += M[row+incRow][column+incColumn];
                            count++;
                        }
                    }
                }
                // Note: Check for zero counts, to prevent divide by zero errors.
                smoother[row][column] = (count > 0) ? (sum / count) : 0;
                //smoother[row][column] = sum / count;
            }
        }
        return smoother;
    }

    /**
     * isWithinBounds(int r, int c, int numRows, int numColumns):
     * Checks if given coordinate (r,c) is within bound of matrix of size [numRows, numColumns]
     * Used as a helper function in imageSmoother2.
     * @param r
     * @param c
     * @param numRows
     * @param numColumns
     * @return
     */
    public static boolean isWithinBounds(int r, int c, int numRows, int numColumns){
        return (r >= 0) && (c >= 0) && (r < numRows) && (c < numColumns);
    }

    public static void printImageContents(int[][] image){
        for(int row = 0; row < image.length; row++){
            for(int column = 0; column < image[0].length; column++){
                System.out.print(image[row][column] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args){
        // Test 1
        int[][] image = new int[][]{{1,1,1},
                                    {1,0,1},
                                    {1,1,1}};
        int[][] smoothImage1 = imageSmoother1(image);
        System.out.println("Output of imageSmoother1:");
        printImageContents(smoothImage1);
        int[][] smoothImage2 = imageSmoother2(image);
        System.out.println("Output of imageSmoother2:");
        printImageContents(smoothImage2);

        // Test 2
        int[][] image2 = new int[][]{{3,4,5,6},
                                     {2,7,9,4},
                                     {1,2,3,4},
                                     {5,6,7,8},
                                     {9,10,11,12}};
        int[][] smoothImage3 = imageSmoother1(image2);
        System.out.println("Output of imageSmoother1:");
        printImageContents(smoothImage3);
        int[][] smoothImage4 = imageSmoother2(image2);
        System.out.println("Output of imageSmoother2:");
        printImageContents(smoothImage4);

    }
}