"""File that contains the Robot Class"""

import math
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_name, header, footer, cols):
    """Loads in data from a text file
       Function to import data from a file

       Input:
            file_name = Full file names
            header = number of header rows starting from 0
            footer = number of footer rows starting from 0
            cols = select the columns with useful data. set to None for all columns

       Output:
            data = list type of data from file.
       Files must be in same directory as the python file."""

    data = np.genfromtxt(file_name, skip_header=header, skip_footer=footer,
                         names=True, dtype=None, delimiter=' ', usecols=cols)
    return data

def plot_map(landmark_data):
    """Plots the Landmarks Locations and Walls

       Input:
             landmark_data: list of landmark_data formatted like the original .dat file

       Output:
              None. Adds Landmark and walls to final plot"""

    # parse landmark data to plot
    _ignore, land_x, land_y, _ignore, _ignore = map(list, zip(*landmark_data))

    plt.plot(land_x, land_y, 'ro', markersize=3)

    # add landmark labels
    for _a, item in enumerate(landmark_data):
        plt.annotate('%s' %item[0], xy=(item[1], item[2]), xytext=(3, 3),
                     textcoords='offset points')

    # Set outer wall locations
    walls_x = [land_x[1], land_x[4], land_x[9], land_x[11], land_x[12], land_x[13], land_x[14],
               land_x[6], land_x[5], land_x[2], land_x[0], land_x[3], land_x[4]]
    walls_y = [land_y[1], land_y[4], land_y[9], land_y[11], land_y[12], land_y[13], land_y[14],
               land_y[6], land_y[5], land_y[2], land_y[0], land_y[3], land_y[4]]

    plt.plot(walls_x, walls_y, 'k')

    # set inner wall locations
    walls_x = [land_x[10], land_x[8], land_x[7]]
    walls_y = [land_y[10], land_y[8], land_y[7]]

    plt.plot(walls_x, walls_y, 'k', label='_nolegend_')

class Robot:
    """"A Class to interact with the motion model of a robot moving in 2D space"""

    def __init__(self):

        self.initial_pos = [0, 0, 0]
        self.motion_path = [[0, 0, 0]]
        self.new_pos = [0, 0, 0]
        self.cur_pos = [0, 0, 0]

    def set_initial_pos(self, starting_point):
        """Clear all position arrays and set new starting point"""

        self.initial_pos = []
        self.motion_path = [[]]
        self.new_pos = []
        self.cur_pos = []

        self.initial_pos = starting_point[:]
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

    def part_a2(self):
        """Part A2 Routine"""

        #create Robot Object
        self.set_initial_pos([0, 0, 0])
        plt.figure()

        #Part A2 Data Set
        movement_data = [[0.5, 0, 1], [0, -1/(2*np.pi), 1], [.5, 0, 1],
                         [0, 1/(2*np.pi), 1], [.5, 0, 1]]

        # loop through data set calulate robot movement using
        # motion model
        for item in enumerate(movement_data):
            # calculate new state
            self.make_move(item[1], 0)

        # parse data to plot
        x_arr, y_arr, _t_arr = map(list, zip(*self.motion_path))


        plt.plot(x_arr, y_arr, 'b')

        plt.title('Part A.2 Results')
        plt.xlabel('Robot X Position (m)')
        plt.ylabel('Robot Y Position (m)')
        plt.legend(['Robot Trajectory'])
        #plt.show()

    def part_a6(self):
        """Part A6 Routine"""

        #create Robot Object
        self.set_initial_pos([0, 0, 0])

        #Part A6 Data Set
        sub_num = [6, 13, 17]
        marker_set = [[1.88032539, -5.57229508], [3.07964257, 0.24942861],
                      [-1.04151642, 2.80020985]]

        robot_locs = [[2, 3, 0], [0, 3, 0], [1, -2, 0]]

        global_results = [0, 0, 0]
        error_calc = [0, 0, 0]

        # Calculate Measurement
        for i, item in enumerate(marker_set):

            # set current poisition
            self.new_pos = robot_locs[i]

            # calculate measurement
            results = self.read_sensor(item, 1)

            # convert measurement to global domain
            global_results[i] = [self.new_pos[0] + np.cos(results[1]) * results[0],
                                 self.new_pos[1] + np.sin(results[1]) * results[0]]

            # calculate error
            error_calc[i] = [marker_set[i][0] - global_results[i][0],
                             marker_set[i][1] - global_results[i][1]]

            # print error calculations to console
            print "For Subject #" + str(sub_num[i]) + ":"
            print "The error in the x direction is " + str(round(error_calc[i][0], 5))
            print "The error in the y direction is " + str(round(error_calc[i][1], 5))

    def part_a3(self):
        """Main Routine"""

    # load in data files

        odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
        meas_data = load_data('ds1_Measurement.dat', 3, 0, [0, 4, 6, 7])
        marker_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])
        bar_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
        groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

    # create Robot Object

        self.set_initial_pos([groundtruth_data[0][1], groundtruth_data[0][2],
                              groundtruth_data[0][3]])

        plt.figure()

    # Transform Measurement Subject from Barcode # to Subject #
        for _counter, item in enumerate(meas_data):
            for _match, match in enumerate(bar_data):
                if match[1] == item[1]:
                    item[1] = match[0]
                    break

    # plot Landmarks and Walls

        plot_map(marker_data)

    # Plot Groundtruth Data

        _ignore, ground_x, ground_y, _ground_t = map(list, zip(*groundtruth_data))

        #ground_x = ground_x[0:20000]
        #ground_y = ground_y[0:20000]

        #plt.plot(ground_x[0], ground_y[0], 'kd', markersize=3, label='_nolegend_')
        plt.plot(ground_x, ground_y, 'g')
        #plt.plot(ground_x[-1], ground_y[-1], 'ko', markersize=3, label='_nolegend_')

    # Plot Odometry Dataset

        for i, cur_action in enumerate(odom_data):

            if i + 1 >= len(odom_data):
                break

            movement_data = [odom_data[i][1], odom_data[i][2], odom_data[i+1][0] - cur_action[0]]
            self.make_move(movement_data, 1)

        x_arr, y_arr, _t_arr = map(list, zip(*self.motion_path))


        plt.plot(x_arr, y_arr, 'b')

        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory'])
        plt.xlabel('World X Position (m)')
        plt.ylabel('World Y Position (m)')
        plt.title('Odometry Data Vs. Groundtruth Data')

        plt.autoscale(True)
        #plt.show()

    def show_plots(self):
        """Show all plots"""
        plt.show()
