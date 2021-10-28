"""
Robot Room Cleaner
"""
# This is the robot's control interface.
class Robot:
   def move(self) -> bool:
       """
       Returns true if the cell in front is open and robot moves into the cell.
       Returns false if the cell in front is blocked and robot stays in the current cell.
       """
       return True

   def turnLeft(self) -> None:
       """
       Robot will stay in the same cell after calling turnLeft/turnRight. Each turn will be 90 degrees.
       """
       pass

   def turnRight(self) -> None:
       """
       Robot will stay in the same cell after calling turnLeft/turnRight. Each turn will be 90 degrees.
       """
       pass

   def clean(self) -> None:
       """
       Clean the current cell.
       """
       pass
from typing import List
class Solution:
    def cleanRoom(self, robot) -> None:
        """
        :type robot: Robot
        :rtype: None
        """
        # Turn the robot back, facing prev direction: R, R, M, R, R
        def goBack():
            robot.turnRight()
            robot.turnRight()
            robot.move()
            robot.turnRight()
            robot.turnRight()

        def backtrack(cell : List[int] = (0,0), dirIndex = 0):
            visited.add(cell)
            robot.clean()
            # Go clockwise: 0: up, 1: right, 2: down, 3: left
            for i in range(4):
                newDirIndex = (dirIndex + i) % 4
                # CAREFUL: Define cell as tuple, not a list, because tuples are hashable,
                # lists are not hashable, and visited set will have hashable cell tuples
                newCell = ( cell[0] + directions[newDirIndex][0], \
                            cell[1] + directions[newDirIndex][1])
                # If cell is not visited and no obstacles, visit
                if not newCell in visited and robot.move():
                    # B5. Backtrack
                    backtrack(newCell, newDirIndex)
                    # B6. Unmake move
                    goBack()
                # Otherwise, cell is visited or hit obstacles
                # Turn the robot following chosen direction: clockwise
                robot.turnRight()

        # Clockwise from up
        directions = [ [-1, 0], [0, 1], [1, 0], [0, -1] ]
        visited = set()
        backtrack()