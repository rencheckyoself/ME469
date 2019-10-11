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

        # load in data files
        self.odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
        self.meas_data = load_data('ds1_Measurement.dat', 3, 0, [0, 4, 6, 7])
        self.marker_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])
        self.bar_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
        self.groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

        self.initial_pos = [self.groundtruth_data[0][1], self.groundtruth_data[0][2],
                            self.groundtruth_data[0][3]]
        self.motion_path = [[0, 0, 0]]
        self.new_pos = [0, 0, 0]
        self.cur_pos = self.initial_pos[:]
        self.M = 100
        self.X_set = []

        self.found_measurements = []

        self.generate_particle_set()
        self.change_measurement_subject()

    def set_initial_pos(self, starting_point):
        """Clear all position arrays and set new starting point"""

        self.initial_pos = []
        self.motion_path = []
        self.new_pos = []
        self.cur_pos = []

        self.initial_pos = starting_point[:]
        self.motion_path = [starting_point[:]]
        self.new_pos = starting_point[:]
        self.cur_pos = starting_point[:]

    def generate_particle_set(self):
        """generate random particle set"""

        k = 0
        print "Generating particle set..."

        while k <= self.M - 1:
            #x,y,theta
            self.X_set.append([np.random.normal(self.initial_pos[0], 0.000009),
                               np.random.normal(self.initial_pos[1], 0.000009),
                               np.random.normal(self.initial_pos[2], 0.003),
                               1/self.M,
                               k])
            k += 1

        # x_arr, y_arr, _h, _h, _h = map(list, zip(*self.X_set))
        #
        # plt.figure()
        # plt.plot(x_arr, y_arr, 'ro', markersize=2)
        # plt.plot(self.initial_pos[0], self.initial_pos[1], 'ko', markersize=5)

    def change_measurement_subject(self):
        """ Transform Measurement Subject from Barcode # to Subject # """

        for counter, item in enumerate(self.meas_data):
            for _match, match in enumerate(self.bar_data):
                if match[1] == item[1]:
                    self.meas_data[counter][1] = match[0]
                    break

    def make_move(self, movement_set, state, noise_check):
        """Used to calculate the motion model
            Function to return the next point of motion over some change in time given
            the starting point and velocities.

            Input Variable Syntax:
              movement_set = [v,w,dt] array with the first three positions
                                      as the control inputs and timestep
              state = [x,y,theta] array with the first three positions as the start vaiables
              noise_check = 1 to turn on noise, else add zero noise

            Output Variable:
              new_pos = [x,y,theta]
              The new position of the robot after some change in time
        """

        vel = [0, 0, 0]

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
        if movement_set[1] == 0:

            vel[0] = movement_set[0]*np.cos(state[2])
            vel[1] = movement_set[0]*np.sin(state[2])
            vel[2] = movement_set[1]

            for i in range(3):
                self.new_pos[i] = state[i] + vel[i] * movement_set[2] + epsilon[i]

        else:

            v_diff = movement_set[0] / movement_set[1]

            vel[0] = -v_diff * np.sin(state[2]) + v_diff * np.sin(state[2] + movement_set[2] * movement_set[1])
            vel[1] = v_diff * np.cos(state[2]) - v_diff * np.cos(state[2] + movement_set[2] * movement_set[1])
            vel[2] = movement_set[1] * movement_set[2]



            for i in range(3):
                self.new_pos[i] = state[i] + vel[i]

    def append_path(self):
        """Add next discrete point to path"""

        self.motion_path.append(self.new_pos[:])

    def read_sensor(self, target_set, body_set, noise_check):
        """Used to calculate the Measurement model
             Function to return the distance and bearing to a landmark given
             the robot location and landmark location.

             Input Variable Syntax:
                    target_set = [sub#, x, y] global location of a landmark
                    body_set = [x,y,theat]

             Output Variable:
                    expected_value = [r, theta] measurement with respect to input"""

        expected_value = [0, 0]

        diff = [0, 0]

        diff[0] = target_set[1] - body_set[0]
        diff[1] = target_set[2] - body_set[1]


        # Set noise value
        # std dev assume to be 0.008m
        if noise_check == 1:
            delta = np.random.normal(0, 0.000064)
        else:
            delta = 0

        # Calculate the distance and bearing to the landmark
        expected_value[0] = round((diff[0]**2 + diff[1]**2)**.5 + delta, 5)
        expected_value[1] = round(math.atan2(diff[1], diff[0]) - body_set[2] + delta, 5)

        return expected_value

    def calc_weight(self, landmark_set, particle_arr, measure_num):
        """ Calulates the weight component of a particle for a given measurement"""

        # feed particle state through sensor model.
        particle_meas = self.read_sensor(landmark_set, particle_arr, 0)

        # calculate error
        error_dist = 1 - (abs(self.found_measurements[measure_num][2] - particle_meas[0]) /
                          self.found_measurements[measure_num][2])

        if ((self.found_measurements[measure_num][3] >= 0 and particle_meas[1] >= 0) or
                (self.found_measurements[measure_num][3] <= 0 and particle_meas[1] <= 0)):

            error_head = 1 - (abs(self.found_measurements[measure_num][3] - particle_meas[1]) /
                              self.found_measurements[measure_num][3])

        else:

            error_head = 1 - (abs(self.found_measurements[measure_num][3]) + abs(particle_meas[1]) /
                              self.found_measurements[measure_num][3])

        # determine weight
        weight = error_dist + error_head

        return weight

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
            self.make_move(item[1], self.cur_pos[:], 0)
            self.cur_pos = self.new_pos[:]

            self.append_path()

        #Plot Data
        x_arr, y_arr, _t_arr = map(list, zip(*self.motion_path))

        plt.plot(x_arr, y_arr, 'b')

        plt.title('Part A.2 Results')
        plt.xlabel('Robot X Position (m)')
        plt.ylabel('Robot Y Position (m)')
        plt.legend(['Robot Trajectory'])

    def part_a6(self):
        """Part A6 Routine"""

        #create Robot Object
        self.set_initial_pos([0, 0, 0])

        #Part A6 Data Set
        marker_set = [[6, 1.88032539, -5.57229508], [13, 3.07964257, 0.24942861],
                      [17, -1.04151642, 2.80020985]]

        robot_locs = [[2, 3, 0], [0, 3, 0], [1, -2, 0]]

        global_results = [0, 0, 0]
        error_calc = [0, 0, 0]

        # Calculate Measurement
        for i, item in enumerate(marker_set):

            # set current poisition
            self.new_pos = robot_locs[i]

            # calculate measurement
            results = self.read_sensor(item, robot_locs[i], 1)

            # convert measurement to global domain
            global_results[i] = [self.new_pos[0] + np.cos(results[1]) * results[0],
                                 self.new_pos[1] + np.sin(results[1]) * results[0]]

            # calculate error
            error_calc[i] = [marker_set[i][1] - global_results[i][0],
                             marker_set[i][2] - global_results[i][1]]

            # print error calculations to console
            print "For Subject #" + str(marker_set[i][0]) + ":"
            print "The measured distance is " + str(results[0]) + "m and the calculated bearing is " + str(results[1]) + "rad."
            print "The error in the x direction is " + str(round(error_calc[i][0], 5))
            print "The error in the y direction is " + str(round(error_calc[i][1], 5))
            print "\n"

    def part_a3(self):
        """Main Routine"""

    # create Robot Object
        self.set_initial_pos([self.groundtruth_data[0][1], self.groundtruth_data[0][2],
                              self.groundtruth_data[0][3]])

        plt.figure()

    # plot Landmarks and Walls
        plot_map(self.marker_data)

    # Plot Groundtruth Data
        _ignore, ground_x, ground_y, _ground_t = map(list, zip(*self.groundtruth_data))

        #ground_x = ground_x[0:20000]
        #ground_y = ground_y[0:20000]

        #plt.plot(ground_x[0], ground_y[0], 'kd', markersize=3, label='_nolegend_')
        plt.plot(ground_x, ground_y, 'g')
        #plt.plot(ground_x[-1], ground_y[-1], 'ko', markersize=3, label='_nolegend_')

    # Plot Odometry Dataset
        for i, cur_action in enumerate(self.odom_data):

            if i + 1 >= len(self.odom_data):
                break

            movement_data = [cur_action[1], cur_action[2],
                             self.odom_data[i+1][0] - cur_action[0]]

            self.make_move(movement_data, self.cur_pos[:], 1)
            self.cur_pos = self.new_pos
            self.append_path()

    # Plot Odometry Data
        x_arr, y_arr, _t_arr = map(list, zip(*self.motion_path))

        plt.plot(x_arr, y_arr, 'b')

        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory'])
        plt.xlabel('World X Position (m)')
        plt.ylabel('World Y Position (m)')
        plt.title('Odometry Data Vs. Groundtruth Data')

        plt.autoscale(True)

    def part_b7(self):
        """Apply particle filter to the data"""

        self.set_initial_pos([self.groundtruth_data[0][1], self.groundtruth_data[0][2],
                              self.groundtruth_data[0][3]])
        last_meas = 0
        # Robot Moves - now move each particle.
        i = 0
        for i, vel_data in enumerate(self.odom_data):
            if i > 472:
                break

            if vel_data[1] == 0 and vel_data[2] == 0:
                pass

            else:

                k = last_meas
                num_measurements = 0
                total_q = 0
                found_markers = []
                self.found_measurements = []

                # check odom timestamp against measurement timestamp and gather measurements
                while k <= len(self.meas_data):

            # save data for later if there is a measurement between my current and future timestep
                    if (self.meas_data[k][0] >= self.odom_data[i][0] and
                            self.meas_data[k][0] < self.odom_data[i+1][0] and
                            self.meas_data[k][1] > 5):

                        num_measurements += 1

                        f = 0

                        while f <= (len(self.marker_data) - 1):

                            if self.marker_data[f][0] == self.meas_data[k][1]:

                                self.found_measurements.append(self.meas_data[k])
                                found_markers.append(self.marker_data[f])
                                break

                            f += 1

                    if self.meas_data[k][0] >= self.odom_data[i+1][0]:

                        last_meas = k
                        break

                    k += 1

                # propigate particles based on control
                print num_measurements
                for j, particle in enumerate(self.X_set):

                    prop_part = [vel_data[1], vel_data[2], self.odom_data[i+1][0] - vel_data[0]]
                    self.make_move(prop_part, particle, 0)

                    # if i >= 470:
                    #     print i, particle

                    particle[0] = self.new_pos[0]
                    particle[1] = self.new_pos[1]
                    particle[2] = self.new_pos[2]
                    particle[3] = 0

                    # if there was a measurement then find particle weight
                    if num_measurements > 0:

                        for l, mark in enumerate(found_markers):

                            particle[3] += self.calc_weight(mark, particle, l)

                        #divide by total measurements
                        particle[3] = particle[3] / num_measurements

                        total_q += particle[3]

                if num_measurements > 0:

                    #resample
                    _ignore, _ignore, _ignore, tot_weights, part_id = list(map(list, zip(*self.X_set)))

                    norm_weights = list(np.divide(tot_weights, total_q))

                    resamp_ids = list(np.random.choice(part_id, self.M, norm_weights))

                    resamp_ids.sort()

                    search = 0
                    new_x_set = []
                    for p, p_id in enumerate(resamp_ids):

                        while search <= len(self.X_set):

                            if self.X_set[search][4] == p_id:
                                new_x_set.append(self.X_set[p])

                            if self.X_set[search][4] > p_id:
                                break

                            search += 1

                    self.X_set = new_x_set[:]


            mean_vals = np.sum(self.X_set, axis=0)
            mean_vals = np.divide(mean_vals, self.M)

            self.new_pos = [mean_vals[0], mean_vals[1], mean_vals[2]]

            self.append_path()

        x_arr, y_arr, _t_arr = map(list, zip(*self.motion_path))

        plt.plot(x_arr, y_arr, 'ro', markersize=2)

    def show_plots(self):
        """Show all plots"""
        plt.show()
