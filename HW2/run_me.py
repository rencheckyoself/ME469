"""Main Run File"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import machine_learning


def main():
    """
    Main Execution Function
    """
    go = HW2(1000, 550) # (N_test, offset) Change to (1000, 3550) for full Test2
                        #                  See Class for a full description

    print("part a 2D Running...")
    part_a_2d(500, .4) #(N, h) - Sine Wave Test
    print("part a 3D Running...")
    part_a_3d(500, .4) #(N, h) - 3D Sine Wave Test

    print("part b Running...")
    go.part_b() # Main ML Function

    data_files = ["x_data.csv", "y_data.csv", "th_data.csv"]
    go.part_b_eval(data_files) # Used create some plots and play with data from a
                               # previous run based on specified data files

    plt.show()

class HW2(object):
    """
    Class to handle all of the dataset manipulation before and after running it
    through the LWR algorithm

    Inputs:
        N_test = the number of datapoints from the learning_data to be set aside
        for testing

        offset = the starting index of where the testing data should be pulled
        from

        Ex. (1000, 3000) The testing data set is 1000 data points starting at
        at index 3000 from the learning_data file

    """
    def __init__(self, N_test, offset):

        self.learned_data = machine_learning.load_data("learning_dataset.csv", 0, 0, None)

        self.N_test = N_test
        self.N_train = self.learned_data.shape[0] - self.N_test
        self.offset = offset
        self.h = .06

    def part_b(self):
        """
        Function to
            1) Create the training set and testing set and format it for LWR Alg
            2) Run test set through LWR
            3) Save output into csv files
            4) Create some plots
        """

        # Used to export data for later use to avoid running multipe times
        x_data_file = open('x_data.csv', 'wb')
        write_x_data = csv.writer(x_data_file, delimiter=" ")

        y_data_file = open('y_data.csv', 'wb')
        write_y_data = csv.writer(y_data_file, delimiter=" ")

        th_data_file = open('th_data.csv', 'wb')
        write_th_data = csv.writer(th_data_file, delimiter=" ")

        test_inputs_file = open('test_input.csv', 'wb')
        write_test_inputs = csv.writer(test_inputs_file, delimiter=" ")

        training_ldata = np.zeros([self.N_train, len(self.learned_data[0])])
        test_ldata = np.zeros([self.N_test, len(self.learned_data[0])])

        # split up total data set for training and testing
        for i, row in enumerate(self.learned_data):
            for j, col in enumerate(row):

                if i < self.offset:
                    training_ldata[i][j] = col

                elif i >= self.offset and i < self.offset + self.N_test:
                    test_ldata[i - self.offset][j] = col

                    if j == len(row)-1:
                        write_test_inputs.writerow(test_ldata[i - self.offset])

                else:
                    arr_row = i - self.N_test
                    training_ldata[arr_row][j] = col


        # Create training data sets for LWLR
        v = np.reshape(training_ldata[0:, 0], (self.N_train, 1))
        w = np.reshape(training_ldata[0:, 1], (self.N_train, 1))
        x = np.reshape(training_ldata[0:, 6], (self.N_train, 1))
        th = np.reshape(training_ldata[0:, 8], (self.N_train, 1))

        dx = training_ldata[0:, 2]
        dy = training_ldata[0:, 3]
        dth = training_ldata[0:, 4]

        # dx training data: v, w, th, 1 -> dx
        train_x_inputs = np.hstack([v, w, th, np.ones([self.N_train, 1])])
        train_x_output = dx

        # dy training data: v, w, th, 1 -> dy
        train_y_inputs = np.hstack([v, w, th, np.ones([self.N_train, 1])])
        train_y_output = dy

        # dth training data: v, w, x, 1 -> dth
        train_th_inputs = np.hstack([v, w, x, np.ones([self.N_train, 1])])
        train_th_output = dth

        # Generate test data set
        v = np.reshape(test_ldata[0:, 0], (self.N_test, 1))
        w = np.reshape(test_ldata[0:, 1], (self.N_test, 1))
        x = np.reshape(test_ldata[0:, 6], (self.N_test, 1))
        th = np.reshape(test_ldata[0:, 8], (self.N_test, 1))

        dx = test_ldata[0:, 2]
        dy = test_ldata[0:, 3]
        dth = test_ldata[0:, 4]

        test_x_inputs = np.hstack([v, w, th, np.ones([self.N_test, 1])])
        test_x_predic = np.zeros([self.N_test, 1])
        test_x_known = dx

        test_y_inputs = np.hstack([v, w, th, np.ones([self.N_test, 1])])
        test_y_predic = np.zeros([self.N_test, 1])
        test_y_known = dy

        test_th_inputs = np.hstack([v, w, x, np.ones([self.N_test, 1])])
        test_th_predic = np.zeros([self.N_test, 1])
        test_th_known = dth


        # Initialize LWLR Object
        ML_x = machine_learning.LWLR(train_x_inputs, train_x_output, self.h)
        ML_y = machine_learning.LWLR(train_y_inputs, train_y_output, self.h)
        ML_th = machine_learning.LWLR(train_th_inputs, train_th_output, self.h)

        var = np.zeros([self.N_test, 3])
        MSE_cv = np.zeros([self.N_test, 3])

        # make predictions for each test data point
        for q in range(self.N_test):
            test_x_predic[q] = ML_x.weighted_prediction(test_x_inputs[q])
            var[q][0], MSE_cv[q][0] = ML_x.evaluate_learning()

            test_y_predic[q] = ML_y.weighted_prediction(test_y_inputs[q])
            var[q][1], MSE_cv[q][1] = ML_y.evaluate_learning()

            test_th_predic[q] = ML_th.weighted_prediction(test_th_inputs[q])
            var[q][2], MSE_cv[q][2] = ML_th.evaluate_learning()

            if q % 20 == 0:
                print q

        # Assemble graph arrays
        for p in range(self.N_test):

            xout = [test_x_inputs[p][0], test_x_inputs[p][1], test_x_inputs[p][2], test_x_predic[p][0]]
            yout = [test_y_inputs[p][0], test_y_inputs[p][1], test_y_inputs[p][2], test_y_predic[p][0]]
            thout = [test_th_inputs[p][0], test_th_inputs[p][1], test_th_inputs[p][2], test_th_predic[p][0]]

            # Save Out Data
            write_x_data.writerow(xout)
            write_y_data.writerow(yout)
            write_th_data.writerow(thout)

        # Plot Figures
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_x_inputs[0:, 0], train_x_inputs[0:, 2], zs=train_x_output, s=1)
        ax.scatter(test_x_inputs[0:, 0], test_x_inputs[0:, 2], c="r", zs=test_x_predic, zorder=5)
        plt.xlabel("v")
        plt.ylabel("th")
        plt.title("LWR for Change in x")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dx")
        ax.set_zlim(-.5, .5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_x_inputs[0:, 0], train_x_inputs[0:, 1], zs=train_x_output, s=1)
        ax.scatter(test_x_inputs[0:, 0], test_x_inputs[0:, 1], c="r", zs=test_x_predic, zorder=5)
        plt.xlabel("v")
        plt.ylabel("w")
        plt.title("LWR for Change in x")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dx")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_y_inputs[0:, 0], train_y_inputs[0:, 2], zs=train_y_output, s=1)
        ax.scatter(test_y_inputs[0:, 0], test_y_inputs[0:, 2], zs=test_y_predic, c="r", zorder=5)
        plt.xlabel("v")
        plt.ylabel("th")
        plt.title("LWR for Change in y")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dy")
        ax.set_zlim(-.5, .5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_y_inputs[0:, 0], train_y_inputs[0:, 1], zs=train_y_output, s=1)
        ax.scatter(test_y_inputs[0:, 0], test_y_inputs[0:, 1], zs=test_y_predic, c="r", zorder=5)
        plt.xlabel("v")
        plt.ylabel("w")
        plt.title("LWR for Change in y")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dy")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_th_inputs[0:, 1], train_th_inputs[0:, 2], zs=train_th_output, s=1)
        ax.scatter(test_th_inputs[0:, 1], test_th_inputs[0:, 2], zs=test_th_predic, c='r', zorder=5)
        plt.xlabel("w")
        plt.ylabel("x")
        plt.title("LWR for Change in Theta")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dth")
        ax.set_zlim(-1, 1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_th_inputs[0:, 0], train_th_inputs[0:, 1], zs=train_th_output, s=1)
        ax.scatter(test_th_inputs[0:, 0], test_th_inputs[0:, 1], zs=test_th_predic, c='r', zorder=5)
        plt.xlabel("v")
        plt.ylabel("w")
        plt.title("LWR for Change in Theta")
        plt.legend(["Training Data", "Testing Data"])
        ax.set_zlabel("dth")
        ax.set_zlim(-5, 5)

        fig = plt.figure()
        plt.plot(test_x_known, 'b')
        plt.plot(test_x_predic, 'r')
        plt.title("dx Comparison")
        plt.xlabel("Query Index")
        plt.ylabel("dx")
        plt.legend(["Known", "Predicted"])

        fig = plt.figure()
        plt.plot(test_y_known, 'b')
        plt.plot(test_y_predic, 'r')
        plt.title("dy Comparison")
        plt.xlabel("Query Index")
        plt.ylabel("dy")
        plt.legend(["Known", "Predicted"])

        fig = plt.figure()
        plt.plot(test_th_known, 'b')
        plt.plot(test_th_predic, 'r')
        plt.title("dth Comparison")
        plt.xlabel("Query Index")
        plt.ylabel("dth")
        plt.legend(["Known", "Predicted"])

        for i in range(3):
            header = ["x", "y", "th"]

            # fig = plt.figure()
            # plt.plot(MSE_cv[0:, i], 'b')
            # plt.title("Test Cross Validation for " + header[i])
            # plt.xlabel("Query Index")
            # plt.ylabel("MSE_cv")

            fig = plt.figure()
            plt.plot(var[0:, i], 'b')
            plt.title("Test Variance for " + header[i])
            plt.xlabel("Query Index")
            plt.ylabel("Varience")



    def part_b_eval(self, data_files):
        """
        Function to do some extra post processing of the results using the saved
        data.

        MAKE SURE THE CLASS WAS INITIALZED WITH N_TEST AND OFFSET VALUES THAT
        CORRESPOND TO THE SAVED DATA.

        Creates the error plots and trjectory plot
        """
        x_results = machine_learning.load_data(data_files[0], 0, 0, None)
        y_results = machine_learning.load_data(data_files[1], 0, 0, None)
        th_results = machine_learning.load_data(data_files[2], 0, 0, None)

        i = self.offset + 1

        gt_arr = np.zeros([self.N_test-2, 2])
        lwlr_arr = np.zeros([self.N_test-1, 2])
        odom_arr = np.zeros([self.N_test-1, 3])

        error_comp = np.zeros([self.N_test, 2])

        lwlr_arr[0][0] = self.learned_data[i][6]
        lwlr_arr[0][1] = self.learned_data[i][7]

        odom_arr[0][0] = self.learned_data[i][6]
        odom_arr[0][1] = self.learned_data[i][7]
        odom_arr[0][2] = self.learned_data[i][8]

        while i < self.N_test + self.offset - 1:

            j = i - (self.offset + 1)
            gt_arr[j][0] = self.learned_data[i][6]
            gt_arr[j][1] = self.learned_data[i][7]

            dt = self.learned_data[i][5]
            dx = x_results[j][3]
            dy = y_results[j][3]

            lwlr_arr[j+1][0] = lwlr_arr[j][0] + dx * dt
            lwlr_arr[j+1][1] = lwlr_arr[j][1] + dy * dt

            movement_set = [self.learned_data[i][0],
                            self.learned_data[i][1],
                            dt]

            cur_state = odom_arr[j][:]

            new_pos = self.motion_model(movement_set, cur_state, 0)

            odom_arr[j+1][0] = new_pos[0]
            odom_arr[j+1][1] = new_pos[1]
            odom_arr[j+1][2] = new_pos[2]

            error_comp[j][0] = abs(dx - self.learned_data[i][2])
            error_comp[j][1] = abs(dy - self.learned_data[i][3])

            i += 1

        plt.figure()

        self.plot_map()

        plt.plot(gt_arr[0:, 0], gt_arr[0:, 1], 'g')
        plt.plot(lwlr_arr[0:, 0], lwlr_arr[0:, 1], 'r')
        plt.plot(odom_arr[0:, 0], odom_arr[0:, 1], 'b')

        plt.title("LWR Performance Comparison")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(['Landmark', 'Wall', 'Groundtruth', 'LWR Trajectory', 'Odometry'])
        plt.xlim([-2, 8])
        plt.ylim([-6, 6])

        plt.figure()
        plt.plot(error_comp[0:, 0], 'b')
        plt.xlabel("Query Index")
        plt.ylabel("Absolute Error")
        plt.title("Error Comparison for change in x")

        plt.figure()
        plt.plot(error_comp[0:, 1], 'b')
        plt.xlabel("Query Index")
        plt.ylabel("Absolute Error")
        plt.title("Error Comparison for change in y")

    def motion_model(self, movement_set, state, noise_check):
        """Used to calculate the motion model
            Function to return the next point of motion over some change in time given
            the starting point and velocities.

            Input Variable Syntax:
              movement_set = [v,w,dt] array with the first three positions
                                      as the control inputs and timestep
              state = [x,y,theta] array with the first three positions as the start vaiables
              noise_check = 1 to turn on noise, else add zero noise

            Output:
              new_pos = [x,y,theta]
              The new position of the robot after some change in time
        """

        vel = [0, 0, 0]
        new_pos = [0, 0, 0]

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
                new_pos[i] = state[i] + vel[i] * movement_set[2] + epsilon[i]

        else:

            v_diff = movement_set[0] / movement_set[1]

            vel[0] = -v_diff * np.sin(state[2]) + v_diff * np.sin(state[2] + movement_set[2] * movement_set[1])
            vel[1] = v_diff * np.cos(state[2]) - v_diff * np.cos(state[2] + movement_set[2] * movement_set[1])
            vel[2] = movement_set[1] * movement_set[2]

            for i in range(3):
                new_pos[i] = state[i] + vel[i]

        return new_pos

    def plot_map(self):
        """\
        Plots the Landmarks Locations and Walls

        Input:
            landmark_data: list of landmark_data formatted like the original .dat file

        Output:
            None. Adds Landmark and walls to final plot
        """

        # parse landmark data to plot

        landmark_data = machine_learning.load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])

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


def part_a_2d(N, h):
    """
    Function to test the performance of a learning algoithm using a sine wave
    """
    noise_var = .25
    N = N # Number of data points
    h_val = h

    # Generate training data for a sine wave
    training_data_x1 = np.array([np.linspace(0, 2*np.pi, N)]).T

    training_data_x = np.hstack([training_data_x1, np.ones([training_data_x1.shape[0], 1])])
    training_data_y = np.zeros(N)

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0, noise_var)

    # Generate test data set
    test_data_x1 = (2*np.pi) * np.random.rand(N, 1)
    test_data_x = np.hstack([test_data_x1, np.ones([test_data_x1.shape[0], 1])])

    test_data_pred_w = np.zeros([N, 1])
    test_data_pred_uw = np.zeros([N, 1])

    # Initialize LWLR Object
    ML = machine_learning.LWLR(training_data_x, training_data_y, h_val)

    var = np.zeros(N)
    MSE_cv = np.zeros(N)

    # make predictions for each test data point
    for i, q in enumerate(test_data_x):

        test_data_pred_w[i] = ML.weighted_prediction(q)
        var[i], MSE_cv[i] = ML.evaluate_learning()
        # print ML.evaluate_learning()
        test_data_pred_uw[i] = ML.unweighted_prediction(q)

    plt.plot(training_data_x[0:, 0], training_data_y, 'bo')
    plt.plot(test_data_x[0:, 0], test_data_pred_w, 'ro')
    plt.plot(test_data_x[0:, 0], test_data_pred_uw, 'go')

    plt.title("LWLR Test and Comparison for h=" + str(h_val))
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

    plt.figure()
    plt.plot(var, 'b')
    plt.title("2D Sine Test Variance for h=" + str(h_val))
    plt.xlabel("Query Index")
    plt.ylabel("Varience")
    plt.xlim([0, N])
    plt.ylim([0, None])

    # plt.figure()
    # plt.plot(MSE_cv, 'b')
    # plt.title("Test Cross Validation for h=" + str(h_val))
    # plt.xlabel("Query Index")
    # plt.ylabel("MSE_cv")
    # plt.xlim([0, N])
    # plt.ylim([0, None])

def part_a_3d(N, h):
    """
    Function to test the performance of a learning algoithm using a 3D
    sine wave
    """
    noise_var = .25
    N = N # Number of data points
    h_val = h

    # Generate training data for a sine wave
    training_data_x1 = np.array([np.linspace(0, 2*np.pi, N)]).T
    training_data_x2 = np.array([np.linspace(0, 5, N)]).T

    training_data_x = np.hstack([training_data_x1, training_data_x2,
                                 np.ones([training_data_x1.shape[0], 1])])
    training_data_y = np.zeros(N)

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0, noise_var)

    # Generate test data set
    test_data_x1 = (2*np.pi) * np.random.rand(N, 1)
    test_data_x2 = 5 * (test_data_x1 / (2*np.pi))
    test_data_x = np.hstack([test_data_x1, test_data_x2, np.ones([test_data_x1.shape[0], 1])])

    test_data_pred_w = np.zeros([N, 1])
    test_data_pred_uw = np.zeros([N, 1])

    # Initialize LWLR Object
    ML = machine_learning.LWLR(training_data_x, training_data_y, h_val)

    var = np.zeros(N)
    MSE_cv = np.zeros(N)

    # make predictions for each test data point
    for i, q in enumerate(test_data_x):
        test_data_pred_w[i] = ML.weighted_prediction(q)
        var[i], MSE_cv[i] = ML.evaluate_learning()
        # print ML.evaluate_learning()
        test_data_pred_uw[i] = ML.unweighted_prediction(q)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(training_data_x[0:, 0], training_data_x[0:, 1], zs=training_data_y)
    ax.scatter(test_data_x[0:, 0], test_data_x[0:, 1], zs=test_data_pred_w)
    ax.scatter(test_data_x[0:, 0], test_data_x[0:, 1], zs=test_data_pred_uw)

    plt.title("LWLR Test and Comparison")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

    fig = plt.figure()
    plt.plot(var, 'b')
    plt.title("3D Sine Test Variance")
    plt.xlabel("Query Index")
    plt.ylabel("Varience")
    plt.xlim([0, N])
    plt.ylim([0, None])

    # fig = plt.figure()
    # plt.plot(MSE_cv, 'b')
    # plt.title("Test Cross Validation")
    # plt.xlabel("Query Index")
    # plt.ylabel("MSE_cv")

main()
