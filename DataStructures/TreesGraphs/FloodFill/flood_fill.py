'''
Leetcode 733: Flood Fill

You are given an image represented by an m x n grid of integers image, where image[i][j] represents the pixel value of the image. You are also given three integers sr, sc, and color. Your task is to perform a flood fill on the image starting from the pixel image[sr][sc].
To perform a flood fill:

1. Begin with the starting pixel and change its color to color.
2. Perform the same process for each pixel that is directly adjacent (pixels that share a side with the original pixel, either horizontally or vertically) and shares the same color as the starting pixel.
3. Keep repeating this process by checking neighboring pixels of the updated pixels and modifying their color if it matches the original color of the starting pixel.
4. The process stops when there are no more adjacent pixels of the original color to update.

Return the modified image after performing the flood fill.
'''
from typing import List

class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        rows, cols = len(image), len(image[0])
        color = image[sr][sc]
        if color == newColor:
            return image

        directions = [[0, 1], [0, -1], [1, 0], [-1, 0]]

        def is_within_bounds(row: int, col: int) -> bool:
            return 0 <= row < rows and 0 <= col < cols

        def dfs(r: int, c: int) -> None:
            if image[r][c] == color:
                image[r][c] = newColor
                # Neighbour check
                for d_r, d_c in directions:
                    new_r, new_c = r + d_r, c + d_c
                    if is_within_bounds(new_r, new_c):
                        dfs(new_r, new_c)

        dfs(sr, sc)
        return image
