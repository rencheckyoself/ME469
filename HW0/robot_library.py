"""File that contains the Robot Class"""

import numpy as np

class Robot:
    """"A Class to interact with the motion model of a robot moving in 2D space"""

    def __init__(self, starting_point):

    #   all of these positions are relative to the robot's domain
        initial_pos = starting_point[:]
        self.motion_path = [starting_point[:]]
        self.new_pos = starting_point[:]
        self.cur_pos = starting_point[:]

    # Function to return the next point of motion over some change in time given
    # the starting point and velocities.
    #
    # Motion model consists of standard kinematics equations:
    #   new_position = initial_position + velocity * deltaT
    #
    # Input Variable Syntax:
    #        startPos = [x,y,heading] relative to the robot's domain
    #        movementSet = [v,w,dt]
    #
    # Output Variable:
    #        newPos = [x,y,heading] relative to the robot's domain
    #
    #        The new position of the robot after some change in time

    def make_move(self, movement_set):
        """Used to calculate the motion model"""

        vel = [0, 0, 0]

        #Calculate directional velocities
        self.cur_pos = self.new_pos[:]

        vel[0] = movement_set[0]*np.cos(self.cur_pos[2])
        vel[1] = movement_set[0]*np.sin(self.cur_pos[2])
        vel[2] = movement_set[1]

        #calculate the new
        for i in range(len(self.cur_pos)):

            self.new_pos[i] = self.cur_pos[i] + vel[i]*movement_set[2]

        self.motion_path.append(self.new_pos[:])
