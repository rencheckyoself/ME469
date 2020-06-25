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
    go = HW3(1500, 0)

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

        self.learned_data = machine_learning.load_data("learning_dataset_vish.csv", 0, 0, None)

        self.N_test = N_test
        self.N_train = self.learned_data.shape[0] - self.N_test
        self.offset = offset
        self.h = 10

        print self.N_test, self.N_train

    def run_test(self):
        """
        Function to
            1) Create the training set and testing set and format it for LWR Alg
            2) Run test set through LWR
            3) Save output into csv files
            4) Create some plots
        """

        x_data_file = open('x_data_vish.csv', 'wb')
        write_x_data = csv.writer(x_data_file, delimiter=" ")

        y_data_file = open('y_data_vish.csv', 'wb')
        write_y_data = csv.writer(y_data_file, delimiter=" ")

        test_inputs_file = open('test_input_vish.csv', 'wb')
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
        x = np.reshape(training_ldata[0:, 2], (self.N_train, 1))
        y = np.reshape(training_ldata[0:, 3], (self.N_train, 1))
        sin_th = np.reshape(training_ldata[0:, 4], (self.N_train, 1))
        cos_th = np.reshape(training_ldata[0:, 5], (self.N_train, 1))

        x_o = training_ldata[0:, 6]
        y_o = training_ldata[0:, 7]
        sin_th_o = training_ldata[0:, 8]
        cos_th_o = training_ldata[0:, 9]

        train_x_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_train, 1])])
        train_x_output = x_o

        train_y_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_train, 1])])
        train_y_output = y_o

        train_sth_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_train, 1])])
        train_sth_output = sin_th_o

        train_cth_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_train, 1])])
        train_cth_output = cos_th_o

        # Generate test data set
        v = np.reshape(test_ldata[0:, 0], (self.N_test, 1))
        w = np.reshape(test_ldata[0:, 1], (self.N_test, 1))
        x = np.reshape(test_ldata[0:, 2], (self.N_test, 1))
        y = np.reshape(test_ldata[0:, 3], (self.N_test, 1))
        sin_th = np.reshape(test_ldata[0:, 4], (self.N_test, 1))
        cos_th = np.reshape(test_ldata[0:, 5], (self.N_test, 1))

        test_x_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_test, 1])])
        test_x_known = test_ldata[0:, 6]
        test_x_predic = np.zeros([self.N_test, 1])

        test_y_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_test, 1])])
        test_y_known = test_ldata[0:, 7]
        test_y_predic = np.zeros([self.N_test, 1])

        test_sth_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_test, 1])])
        test_sth_known = test_ldata[0:, 8]
        test_sth_predic = np.zeros([self.N_test, 1])

        test_cth_inputs = np.hstack([v, w, x, y, sin_th, cos_th, np.ones([self.N_test, 1])])
        test_cth_known = test_ldata[0:, 9]
        test_cth_predic = np.zeros([self.N_test, 1])

        # Initialize LWLR Object
        ML_x = machine_learning.LWLR(train_x_inputs, train_x_output, self.h)
        ML_y = machine_learning.LWLR(train_y_inputs, train_y_output, self.h)
        ML_sth = machine_learning.LWLR(train_sth_inputs, train_sth_output, self.h)
        ML_cth = machine_learning.LWLR(train_cth_inputs, train_cth_output, self.h)

        var = np.zeros([self.N_test, 4])
        MSE_cv = np.zeros([self.N_test, 4])
        error_comp = np.zeros([self.N_test, 4])

        tot_x_err = 0
        tot_y_err = 0
        tot_sth_err = 0
        tot_cth_err = 0

        ss_tot_x = 0
        ss_tot_y = 0
        ss_tot_cth = 0
        ss_tot_sth = 0

        mean_x = np.sum(test_x_known)/self.N_test
        mean_y = np.sum(test_y_known)/self.N_test
        mean_cth = np.sum(test_cth_known)/self.N_test
        mean_sth = np.sum(test_sth_known)/self.N_test

        # make predictions for each test data point
        for q in range(self.N_test):
            test_x_predic[q] = ML_x.weighted_prediction(test_x_inputs[q])
            var[q][0], MSE_cv[q][0] = ML_x.evaluate_learning()

            test_y_predic[q] = ML_y.weighted_prediction(test_y_inputs[q])
            var[q][1], MSE_cv[q][1] = ML_y.evaluate_learning()

            test_sth_predic[q] = ML_sth.weighted_prediction(test_sth_inputs[q])
            var[q][2], MSE_cv[q][2] = ML_sth.evaluate_learning()

            test_cth_predic[q] = ML_cth.weighted_prediction(test_cth_inputs[q])
            var[q][3], MSE_cv[q][3] = ML_cth.evaluate_learning()

            error_comp[q][0] = (test_x_predic[q] - test_x_known[q])**2
            tot_x_err += error_comp[q][0]

            error_comp[q][1] = (test_y_predic[q] - test_y_known[q])**2
            tot_y_err += error_comp[q][1]

            error_comp[q][2] = (test_sth_predic[q] - test_sth_known[q])**2
            tot_sth_err += error_comp[q][2]

            error_comp[q][3] = (test_cth_predic[q] - test_cth_known[q])**2
            tot_cth_err += error_comp[q][3]

            ss_tot_x += (test_x_known[q] - mean_x)**2
            ss_tot_y += (test_y_known[q] - mean_y)**2
            ss_tot_cth += (test_cth_known[q] - mean_cth)**2
            ss_tot_sth += (test_sth_known[q] - mean_sth)**2

            if q % 20 == 0:
                print q

        # # Assemble graph arrays
        # for p in range(self.N_test):
        #
        #     xout = [test_x_inputs[p][0], test_x_inputs[p][1], test_x_inputs[p][2], test_x_inputs[p][3], test_x_inputs[p][4], test_x_predic[p][0]]
        #     yout = [test_y_inputs[p][0], test_y_inputs[p][1], test_y_inputs[p][2], test_y_inputs[p][3], test_y_inputs[p][4], test_y_predic[p][0]]
        #
        #     # Save Out Data
        #     write_x_data.writerow(xout)
        #     write_y_data.writerow(yout)

        avg_x_err = tot_x_err/self.N_test
        avg_y_err = tot_y_err/self.N_test
        avg_sth_err = tot_sth_err/self.N_test
        avg_cth_err = tot_cth_err/self.N_test

        R2_x = 1 - (tot_x_err/ss_tot_x)
        R2_y = 1 - (tot_y_err/ss_tot_y)
        R2_cth = 1 - (tot_cth_err/ss_tot_cth)
        R2_sth = 1 - (tot_sth_err/ss_tot_sth)

        print("Avg X Error:")
        print(avg_x_err)

        print("Avg Y Error")
        print(avg_y_err)

        print("Avg sTh Error")
        print(avg_sth_err)

        print("Avg cTh Error")
        print(avg_cth_err)

        print("R2 Values (x, y, cth, sth):")
        print(R2_x)
        print(R2_y)
        print(R2_cth)
        print(R2_sth)

        # Plot Figures
        fig = plt.figure()
        plt.plot(test_x_predic[0:, 0], 'b')
        plt.plot(test_x_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Predicted x")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for X Position")

        fig = plt.figure()
        plt.plot(var[0:, 0], 'b')
        plt.xlabel("Query")
        plt.ylabel("Predicted X Variance")
        plt.title("Varience for X Position")

        fig = plt.figure()
        plt.plot(test_y_predic[0:, 0], 'b')
        plt.plot(test_y_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Predicted y")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for Y Position")

        fig = plt.figure()
        plt.plot(var[0:, 1], 'b')
        plt.xlabel("Query")
        plt.ylabel("Predicted Y Variance")
        plt.title("Varience for Y Position")

        fig = plt.figure()
        plt.plot(test_sth_predic[0:, 0], 'b')
        plt.plot(test_sth_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Predicted Sin(th)")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for Sin(th)")

        fig = plt.figure()
        plt.plot(var[0:, 2], 'b')
        plt.xlabel("Query")
        plt.ylabel("Predicted Sin(th) Variance")
        plt.title("Varience for Sin(th)")

        fig = plt.figure()
        plt.plot(test_cth_predic[0:, 0], 'b')
        plt.plot(test_cth_known, 'r')
        plt.xlabel("Query")
        plt.ylabel("Predicted Cos(th)")
        plt.legend(['prediction', 'actual'])
        plt.title("Predictions for Cos(th)")

        fig = plt.figure()
        plt.plot(var[0:, 3], 'b')
        plt.xlabel("Query")
        plt.ylabel("Predicted Cos(th) Variance")
        plt.title("Varience for Cos(th)")

        fig = plt.figure()
        plt.plot(test_x_predic[0:, 0], test_y_predic[0:, 0], 'b')
        plt.plot(test_x_known, test_y_known, 'r')
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.legend(['prediction', 'actual'])
        plt.title("Trajectory")
        plt.xlim([-1.5,4.5])
        plt.ylim([-5,5])

main()
