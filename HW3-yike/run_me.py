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
    # go = HW2(1000, 550) # (N_test, offset) Change to (1000, 3550) for full Test2
    #                    #                  See Class for a full description
    #
    # print("part a 2D Running...")
    # part_a_2d(500, .4) #(N, h) - Sine Wave Test
    # print("part a 3D Running...")
    # part_a_3d(500, .4) #(N, h) - 3D Sine Wave Test
    #
    # print("part b Running...")
    # go.part_b() # Main ML Function
    #
    # data_files = ["x_data_mjr.csv", "y_data_mjr.csv", "th_data_mjr.csv"]
    # go.part_b_eval(data_files) # Used create some plots and play with data from a
    #                            # previous run based on specified data files

    go = HW3(20, 20)

    go.run_test()

    plt.show()

class HW3(object):
    """
    Class to handle all of the dataset manipulation before and after running it
    through the LWR algorithm for a partners dataset

    Inputs:
        N_test = the number of datapoints from the learning_data to be set aside
        for testing

        offset = the starting index of where the testing data should be pulled
        from

        Ex. (1000, 3000) The testing data set is 1000 data points starting at
        at index 3000 from the learning_data file

    """
    def __init__(self, N_test, offset):

        self.learned_data = machine_learning.load_data("learning_dataset_yike.txt", 0, 0, None)

        self.N_test = N_test
        self.N_train = self.learned_data.shape[0] - self.N_test
        self.offset = offset
        self.h = .06

        print self.N_test, self.N_train

    def run_test(self):
        """
        Function to
            1) Create the training set and testing set and format it for LWR Alg
            2) Run test set through LWR
            3) Save output into csv files
            4) Create some plots
        """

        x_data_file = open('x_data_yike.csv', 'wb')
        write_x_data = csv.writer(x_data_file, delimiter=" ")

        y_data_file = open('y_data_yike.csv', 'wb')
        write_y_data = csv.writer(y_data_file, delimiter=" ")

        test_inputs_file = open('test_input_yike.csv', 'wb')
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
        t = np.reshape(training_ldata[0:, 0], (self.N_train, 1))
        x = np.reshape(training_ldata[0:, 1], (self.N_train, 1))
        y = np.reshape(training_ldata[0:, 2], (self.N_train, 1))
        th = np.reshape(training_ldata[0:, 3], (self.N_train, 1))
        l_x = np.reshape(training_ldata[0:, 6], (self.N_train, 1))
        l_y = np.reshape(training_ldata[0:, 7], (self.N_train, 1))

        r = training_ldata[0:, 4]
        b = training_ldata[0:, 5]

        # l_x training data: x, y, th, l_x, l_y -> r
        train_r_inputs = np.hstack([x, y, th, l_x, l_y, np.ones([self.N_train, 1])])
        train_r_output = r

        # l_y training data: x, y, th, l_x, l_y -> b
        train_b_inputs = np.hstack([x, y, th, l_x, l_y, np.ones([self.N_train, 1])])
        train_b_output = b

        # Generate test data set
        t = np.reshape(test_ldata[0:, 0], (self.N_test, 1))
        x = np.reshape(test_ldata[0:, 1], (self.N_test, 1))
        y = np.reshape(test_ldata[0:, 2], (self.N_test, 1))
        th = np.reshape(test_ldata[0:, 3], (self.N_test, 1))
        l_x = np.reshape(test_ldata[0:, 6], (self.N_test, 1))
        l_y = np.reshape(test_ldata[0:, 7], (self.N_test, 1))

        test_r_inputs = np.hstack([x, y, th, l_x, l_y, np.ones([self.N_test, 1])])
        test_r_known = test_ldata[0:, 4]
        test_r_predic = np.zeros([self.N_test, 1])

        test_b_inputs = np.hstack([x, y, th, l_x, l_y, np.ones([self.N_test, 1])])
        test_b_known = test_ldata[0:, 5]
        test_b_predic = np.zeros([self.N_test, 1])

        # Initialize LWLR Object
        ML_r = machine_learning.LWLR(train_r_inputs, train_r_output, self.h)
        ML_b = machine_learning.LWLR(train_b_inputs, train_b_output, self.h)

        var = np.zeros([self.N_test, 2])
        MSE_cv = np.zeros([self.N_test, 2])

        # make predictions for each test data point
        for q in range(self.N_test):
            test_r_predic[q] = ML_r.weighted_prediction(test_r_inputs[q])
            var[q][0], MSE_cv[q][0] = ML_r.evaluate_learning()

            test_b_predic[q] = ML_b.weighted_prediction(test_b_inputs[q])
            var[q][1], MSE_cv[q][1] = ML_b.evaluate_learning()

            if q % 20 == 0:
                print q

        # Assemble graph arrays
        # for p in range(self.N_test):
        #
        #     xout = [test_x_inputs[p][0], test_x_inputs[p][1], test_x_inputs[p][2], test_x_inputs[p][3], test_x_inputs[p][4], test_x_predic[p][0]]
        #     yout = [test_y_inputs[p][0], test_y_inputs[p][1], test_y_inputs[p][2], test_y_inputs[p][3], test_y_inputs[p][4], test_y_predic[p][0]]
        #
        #     # Save Out Data
        #     write_x_data.writerow(xout)
        #     write_y_data.writerow(yout)


        # Plot Figures
        fig = plt.figure()
        plt.plot(test_r_predic[0:, 0], 'b')
        plt.plot(test_r_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Range")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for Landmark 14 Range Measurments")

        fig = plt.figure()
        plt.plot(test_b_predic[0:, 0], 'b')
        plt.plot(test_b_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Bearing")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for Landmark 14 Bearing Measurments")


main()
