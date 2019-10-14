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
        self.M = 1000
        self.X_set = []
        self.last_id = 0

        self.found_measurements = []

        self.part_a2_path = []
        self.part_a3_path = []
        self.part_b7_path = []
        self.groundtruth_path = []

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

        np.random.seed(40)

        while k <= self.M - 1:
            #x,y,theta, .003,.003,.05
            self.X_set.append([np.random.normal(self.initial_pos[0], .003),
                               np.random.normal(self.initial_pos[1], .003),
                               np.random.normal(self.initial_pos[2], .05),
                               1/float(self.M),
                               self.last_id])
            k += 1

        self.last_id = k

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

        # Create Noise array
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
        """ Calulates the weight component of a particle for a given measurement

        Inputs: landmark_set [sub#, x, y]
                particle_arr [x, y, theta, weight, id]
                measure_num [int] - used to determine which row from measurement
                                    data set should be used.

        Output: w_i [float] probability of the measurement given the particle state and measurement
        """

        # feed particle state through sensor model.
        particle_meas = self.read_sensor(landmark_set, particle_arr, 1)

        dis_error = (particle_meas[0] - self.found_measurements[measure_num][2])**2
        head_error = (particle_meas[1] - self.found_measurements[measure_num][3])**2

        w_i = 1/(dis_error + head_error)

        # co_var = np.matrix([[.003, .05], [.05, .003]])
        # co_varI = co_var.I
        #
        # particle_meas = np.array(particle_meas)
        # meas_arr = np.array([self.found_measurements[measure_num][2],
        #                      self.found_measurements[measure_num][3]])
        #
        # diff_arr = particle_meas - meas_arr
        #
        # diff_arr_t = diff_arr.reshape(2, 1)
        #
        # w_i = float((np.linalg.det(2 * np.pi * co_var)**.5) * np.exp(-.5 * diff_arr * co_varI * diff_arr_t))

        return w_i

    def low_var_resample(self):
        """
        Low Varience Resampling Routine

        Algorithm from Probabilistic Robotics Ch.4 pg.86

        """
        new_x_set = self.X_set[:]

        r = np.random.random() * 1 / float(self.M)
        c = self.X_set[0][3]
        I = 0

        # 20, .00001, 1.5
        for m in range(self.M):
            if m % 20 == 0:
                new_x_set[m] = [np.random.normal(self.cur_pos[0], .00001),
                                np.random.normal(self.cur_pos[1], .00001),
                                np.random.normal(self.cur_pos[2], 1.5),
                                1/(float(self.M)),
                                self.last_id]

                self.last_id += 1

            else:
                u = r + (m / (float(self.M) - 1.0))

                while u > c:
                    I += 1
                    if I > self.M-1:
                        I = self.M-1
                        break
                    c += self.X_set[I][3]
                    #print u, c, I

                new_x_set[m][:] = self.X_set[I][:]


        _ig, _ig, _ig, w_b, _ig = map(list, zip(*new_x_set))

        norm = np.sum(w_b)

        for a in range(len(new_x_set)):
            new_x_set[a][3] = new_x_set[a][3] / norm

        return new_x_set[:]

    def part_a2(self):
        """Part A2 Routine"""

        #create Robot Object
        self.set_initial_pos([0, 0, 0])

        #Part A2 Data Set
        movement_data = [[0.5, 0, 1], [0, -1/(2*np.pi), 1], [.5, 0, 1],
                         [0, 1/(2*np.pi), 1], [.5, 0, 1]]

        # loop through data set calulate robot movement using
        # motion model
        for item in enumerate(movement_data):
            # calculate new state
            self.make_move(item[1], self.cur_pos[:], 1)
            self.cur_pos = self.new_pos[:]

            self.append_path()

        self.part_a2_path = self.motion_path[:]

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

    # Plot Odometry Dataset
        for i, cur_action in enumerate(self.odom_data):

            if i + 1 >= len(self.odom_data):
                break

            movement_data = [cur_action[1], cur_action[2],
                             self.odom_data[i+1][0] - cur_action[0]]

            self.make_move(movement_data, self.cur_pos[:], 1)
            self.cur_pos = self.new_pos
            self.append_path()

        self.part_a3_path = self.motion_path[:]

    def part_b7(self):
        """Apply particle filter to the data"""

        self.set_initial_pos([self.groundtruth_data[0][1], self.groundtruth_data[0][2],
                              self.groundtruth_data[0][3]])
        last_meas = 0
        total_measurements = 0
        num_measurements = 0

        dt_thresh = 1
        iter_thresh = len(self.odom_data) - 2
        delay_thresh = 0
        resample_delay = 0

        i = 0
        for i, vel_data in enumerate(self.odom_data):
            # if i % (iter_thresh/4) == 0:
            #     print i
            if i > iter_thresh:
                break

            if vel_data[1] == 0 and vel_data[2] == 0:
                ## modify to alway propigate particles once bugs worked out.
                pass

            else:

                k = last_meas
                num_measurements = 0
                tot_w = 0
                found_markers = []
                self.found_measurements = []

                # save data for later if there is a measurement between my current and
                # future timestep threshold

                while k <= len(self.meas_data)-1:

                    # check measurement timestamps against odom timestamp
                    if i >= len(self.odom_data)-4:
                        dt_thresh = 0
                    if (self.meas_data[k][0] >= self.odom_data[i][0] and
                            self.meas_data[k][0] < self.odom_data[i+dt_thresh][0] and
                            self.meas_data[k][1] > 5):

                        num_measurements += 1
                        total_measurements += 1

                        f = 0

                        #search for the marker data set corresponding to the measurement found
                        while f <= (len(self.marker_data) - 1):

                            #add marker set to the temporary list
                            if self.marker_data[f][0] == self.meas_data[k][1]:

                                self.found_measurements.append(self.meas_data[k])
                                found_markers.append(self.marker_data[f])
                                break

                            f += 1

                    if self.meas_data[k][0] >= self.odom_data[i+dt_thresh][0]:

                        last_meas = k
                        break

                    k += 1

                if num_measurements > 0:
                    resample_delay += 1

                # propigate each particle based on control
                for j in range(len(self.X_set)):

                    prop_part = [vel_data[1], vel_data[2], self.odom_data[i+1][0] - vel_data[0]]
                    self.make_move(prop_part, self.X_set[j], 1)

                    self.X_set[j][0] = self.new_pos[0]
                    self.X_set[j][1] = self.new_pos[1]
                    self.X_set[j][2] = self.new_pos[2]
                    self.X_set[j][3] = 1 / float(self.M)

                    if num_measurements > 0:

                        for l, mark in enumerate(found_markers):
                            self.X_set[j][3] += self.calc_weight(mark, self.X_set[j], l)

                        #divide by total measurements
                        self.X_set[j][3] = self.X_set[j][3] / num_measurements

                        tot_w += self.X_set[j][3]

                if num_measurements > 0 and resample_delay > delay_thresh:
                    #resample

                    for _a, adjust in enumerate(self.X_set):
                        adjust[3] = adjust[3] / tot_w
                        #print i, self.X_set[a][4], self.X_set[a][3]

                    self.X_set = self.low_var_resample()

                    resample_delay = 0

            # find_max = 0
            # for max_i, values in enumerate(self.X_set):
            #     if find_max < values[3]:
            #         find_max = values[3]
            #         max_index = max_i
            #
            # plt.plot(self.X_set[max_index][0], self.X_set[max_index][1], 'ko', markersize=3)

            # x_arr, y_arr, _h, _h, _h = map(list, zip(*self.X_set))
            # plt.plot(x_arr, y_arr, 'ko', markersize=1)

            mean_x = 0
            mean_y = 0
            mean_th = 0

            #Get mean weighted average for the plotting points
            for _t, avg in enumerate(self.X_set):
                mean_x = mean_x + avg[0] * avg[3]
                mean_y = mean_y + avg[1] * avg[3]
                mean_th = mean_th + avg[2] * avg[3]

            #print mean_x, mean_y, mean_th
            self.new_pos = [mean_x, mean_y, mean_th]
            self.cur_pos = [mean_x, mean_y, mean_th]

            self.append_path()

        self.part_b7_path = self.motion_path[:]
        print "total measurements found: " + str(total_measurements)

    def create_plots(self):
        """
        Create Plots for Report
        """
        #part_a2
        x_arr, y_arr, _t_arr = map(list, zip(*self.part_a2_path))

        plt.figure()
        plt.plot(x_arr, y_arr, 'b')

        plt.title('Part A.2 Results')
        plt.xlabel('Robot X Position (m)')
        plt.ylabel('Robot Y Position (m)')
        plt.legend(['Robot Trajectory'])
        plt.autoscale(True)

        #part 3a
        plt.figure()

        plot_map(self.marker_data)

        _ig, ground_x, ground_y, _ground_t = map(list, zip(*self.groundtruth_data))

        # ground_x = ground_x[0:20000]
        # ground_y = ground_y[0:20000]

        plt.plot(ground_x, ground_y, 'g')

        x_arr, y_arr, _t_arr = map(list, zip(*self.part_a3_path))

        plt.plot(x_arr, y_arr, 'b')

        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory'])
        plt.xlabel('World X Position (m)')
        plt.ylabel('World Y Position (m)')
        plt.title('Odometry Data Vs. Groundtruth Data')
        plt.autoscale(True)

        #part 7b - trunc.

        plt.figure()

        plot_map(self.marker_data)

        _ig, ground_x, ground_y, _ground_t = map(list, zip(*self.groundtruth_data))

        ground_x = ground_x[0:20000]
        ground_y = ground_y[0:20000]

        plt.plot(ground_x, ground_y, 'g')

        x_arr, y_arr, _t_arr = map(list, zip(*self.part_a3_path))

        x_arr = x_arr[0:1700]
        y_arr = y_arr[0:1700]

        plt.plot(x_arr, y_arr, 'b')

        x_arr, y_arr, _t_arr = map(list, zip(*self.part_b7_path))

        x_arr = x_arr[1:1700]
        y_arr = y_arr[1:1700]

        plt.plot(x_arr, y_arr, 'r')

        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory', 'Filter Trajectory'])
        plt.xlabel('World X Position (m)')
        plt.ylabel('World Y Position (m)')
        plt.title('Filter Performance Comparison')
        plt.autoscale(True)

        #7b Full
        plt.figure()

        plot_map(self.marker_data)

        _ig, ground_x, ground_y, _ground_t = map(list, zip(*self.groundtruth_data))

        # ground_x = ground_x[0:20000]
        # ground_y = ground_y[0:20000]

        plt.plot(ground_x, ground_y, 'g')

        x_arr, y_arr, _t_arr = map(list, zip(*self.part_a3_path))

        # x_arr = x_arr[0:1700]
        # y_arr = y_arr[0,1700]

        plt.plot(x_arr, y_arr, 'b')

        x_arr, y_arr, _t_arr = map(list, zip(*self.part_b7_path))

        # x_arr = x_arr[1:1700]
        # y_arr = y_arr[1:1700]

        plt.plot(x_arr, y_arr, 'r')

        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory', 'Filter Trajectory'])
        plt.xlabel('World X Position (m)')
        plt.ylabel('World Y Position (m)')
        plt.title('Filter Performance Comparison')
        plt.autoscale(True)

        plt.show()
