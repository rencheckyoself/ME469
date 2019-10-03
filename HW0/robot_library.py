"""File that contains the Robot Class"""

import math
import numpy as np


class Robot:
    """"A Class to interact with the motion model of a robot moving in 2D space"""

    def __init__(self, starting_point):

    #   all of these positions are relative to the robot's domain
        initial_pos = starting_point[:]
        self.motion_path = [starting_point[:]]
        self.new_pos = starting_point[:]
        self.cur_pos = starting_point[:]



    def make_move(self, movement_set, noise_check):
        """Used to calculate the motion model
            Function to return the next point of motion over some change in time given
            the starting point and velocities.

            Input Variable Syntax:
                  movementSet = [v,w,dt]
                  noise_check = 1 to turn on noise, else add zero noise

            Output Variable:
                  newPos = [x,y,heading] relative to the robot's domain
                  The new position of the robot after some change in time"""

        vel = [0, 0, 0]

        #Calculate directional velocities
        self.cur_pos = self.new_pos[:]

        vel[0] = movement_set[0]*np.cos(self.cur_pos[2])
        vel[1] = movement_set[0]*np.sin(self.cur_pos[2])
        vel[2] = movement_set[1]

        # Create Noise Matrix
        # std dev for x and y assumed to be .004m
        # std dev for theta assumed to be .05rad
        trans_var = 0.000016
        ang_var = 0.0025

        if noise_check == 1:
            epsilon = [np.random.normal(0, trans_var), np.random.normal(0, trans_var),
                       np.random.normal(0, ang_var)]
        else:
            epsilon = [0, 0, 0]

        #calculate the new position
        for i in range(len(self.cur_pos)):

            self.new_pos[i] = self.cur_pos[i] + vel[i]*movement_set[2] + epsilon[i]

        self.motion_path.append(self.new_pos[:])


    def read_sensor(self, target_set, noise_check):
        """Used to calculate the Measurement model
             Function to return the distance and bearing to a landmark given
             the robot location and landmark location.

             Input Variable Syntax:
                    target_set = [x, y] global location of a landmark

             Output Variable:
                    expected_value = [x, y] relative to the robot's location"""

        expected_value = [0, 0]

        diff = [0, 0]

        diff[0] = target_set[0] - self.new_pos[0]
        diff[1] = target_set[1] - self.new_pos[1]


        # Set noise value
        # std dev assume to be 0.009m
        if noise_check == 1:
            delta = np.random.normal(0, 0.000036)
        else:
            delta = 0

        # Calculate the distance and bearing to the landmark
        expected_value[0] = (diff[0]**2 + diff[1]**2)**.5 + delta
        expected_value[1] = math.atan2(diff[1], diff[0]) - self.new_pos[2] + delta

        return expected_value
