"""Main Run File"""

import machine_learning
import numpy as np
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def part_a_2D():
    """
    Function to test the performance of a learning algoithm using a sine wave
    """
    noise_var = .25
    N = 500 # Number of data points
    h_val = 1

    # Generate training data for a sine wave
    training_data_x1 = np.array([np.linspace(0, 2*np.pi, N)]).T

    training_data_x = np.hstack([training_data_x1, np.ones([training_data_x1.shape[0], 1])])
    training_data_y = np.zeros(N)

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0, noise_var)

    # Generate test data set
    test_data_x1 = (2*np.pi) * np.random.rand(N,1)
    test_data_x = np.hstack([test_data_x1, np.ones([test_data_x1.shape[0],1])])

    test_data_pred_w = np.zeros([N,1])
    test_data_pred_uw = np.zeros([N,1])

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

    plt.plot(training_data_x[0:,0], training_data_y, 'bo')
    plt.plot(test_data_x[0:,0], test_data_pred_w, 'ro')
    plt.plot(test_data_x[0:,0], test_data_pred_uw, 'go')

    plt.title("LWLR Test and Comparison for h=" + str(h_val))
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

    fig = plt.figure()
    plt.plot(var, 'b')
    plt.title("Test Variance for h=" + str(h_val))
    plt.xlabel("Query Index")
    plt.ylabel("Varience")
    plt.xlim([0, N])
    plt.ylim([0, None])

    fig = plt.figure()
    plt.plot(MSE_cv, 'b')
    plt.title("Test Cross Validation for h=" + str(h_val))
    plt.xlabel("Query Index")
    plt.ylabel("MSE_cv")
    plt.xlim([0, N])
    plt.ylim([0, None])

def part_a_3D():
    """
    Function to test the performance of a learning algoithm using a sine wave
    """
    noise_var = .25
    N = 500 # Number of data points

    # Generate training data for a sine wave
    training_data_x1 = np.array([np.linspace(0, 2*np.pi, N)]).T
    training_data_x2 = np.array([np.linspace(0, 5, N)]).T

    training_data_x = np.hstack([training_data_x1, training_data_x2, np.ones([training_data_x1.shape[0], 1])])
    training_data_y = np.zeros(N)

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0, noise_var)

    # Generate test data set
    test_data_x1 = (2*np.pi) * np.random.rand(N,1)
    test_data_x2 = 5 * (test_data_x1 / (2*np.pi))
    test_data_x = np.hstack([test_data_x1, test_data_x2, np.ones([test_data_x1.shape[0],1])])

    test_data_pred_w = np.zeros([N,1])
    test_data_pred_uw = np.zeros([N,1])

    # Initialize LWLR Object
    ML = machine_learning.LWLR(training_data_x, training_data_y, 0.4)

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
    ax.scatter(training_data_x[0:,0], training_data_x[0:,1], zs=training_data_y)
    ax.scatter(test_data_x[0:,0], test_data_x[0:,1], zs=test_data_pred_w)
    ax.scatter(test_data_x[0:,0], test_data_x[0:,1], zs=test_data_pred_uw)

    plt.title("LWLR Test and Comparison")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

    fig = plt.figure()
    plt.plot(var, 'b')
    plt.title("Test Variance")
    plt.xlabel("Query Index")
    plt.ylabel("Varience")

    fig = plt.figure()
    plt.plot(MSE_cv, 'b')
    plt.title("Test Cross Validation")
    plt.xlabel("Query Index")
    plt.ylabel("MSE_cv")


class HW2(object):
    def __init__(self, N_test, offset):

        self.learned_data = machine_learning.load_data("learning_dataset.csv", 0, 0, None)
        self.N_test = N_test
        self.N_train = self.learned_data.shape[0] - self.N_test
        self.offset = offset # int(self.learned_data.shape[0] * np.random.rand(1)
        self.h = .06

    def part_b(self):

        x_data_file = open('x_data.csv', 'wb')
        write_x_data = csv.writer(x_data_file, delimiter=" ")

        y_data_file = open('y_data.csv', 'wb')
        write_y_data = csv.writer(y_data_file, delimiter=" ")

        th_data_file = open('th_data.csv', 'wb')
        write_th_data = csv.writer(th_data_file, delimiter=" ")

        test_inputs_file = open('test_input.csv', 'wb')
        write_test_inputs = csv.writer(test_inputs_file, delimiter=" ")

        # split up total data set for training and testing
        training_ldata = np.zeros([self.N_train, len(self.learned_data[0])])
        test_ldata = np.zeros([self.N_test, len(self.learned_data[0])])

        # convert to arrays and split data
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
        v  = np.reshape(training_ldata[0:, 0], (self.N_train, 1))
        w  = np.reshape(training_ldata[0:, 1], (self.N_train, 1))
        x  = np.reshape(training_ldata[0:, 6], (self.N_train, 1))
        th = np.reshape(training_ldata[0:, 8], (self.N_train, 1))

        dx  = training_ldata[0:, 2]
        dy  = training_ldata[0:, 3]
        dth = training_ldata[0:, 4]

        # Change in x training data v, w, th, 1 -> dx
        train_x_inputs = np.hstack([v, w, th, np.ones([self.N_train, 1])])
        train_x_output = dx

        # Change in y training data  v, w, th, 1 -> dy
        train_y_inputs = np.hstack([v, w, th, np.ones([self.N_train, 1])])
        train_y_output = dy

        # Change in th training data  v, w, x, 1 -> dth
        train_th_inputs = np.hstack([v, w, x, np.ones([self.N_train, 1])])
        train_th_output = dth

        # Generate test data set
        v  = np.reshape(test_ldata[0:, 0], (self.N_test, 1))
        w  = np.reshape(test_ldata[0:, 1], (self.N_test, 1))
        x  = np.reshape(test_ldata[0:, 6], (self.N_test, 1))
        th = np.reshape(test_ldata[0:, 8], (self.N_test, 1))

        test_x_inputs = np.hstack([v, w, th, np.ones([self.N_test, 1])])
        test_x_predic = np.zeros([self.N_test, 1])

        test_y_inputs = np.hstack([v, w, th, np.ones([self.N_test, 1])])
        test_y_predic = np.zeros([self.N_test, 1])

        test_th_inputs = np.hstack([v, w, x, np.ones([self.N_test, 1])])
        test_th_predic = np.zeros([self.N_test, 1])

        # Initialize LWLR Object
        ML_x = machine_learning.LWLR(train_x_inputs, train_x_output, self.h)
        ML_y = machine_learning.LWLR(train_y_inputs, train_y_output, self.h)
        ML_th = machine_learning.LWLR(train_th_inputs, train_th_output, self.h)

        var = np.zeros([self.N_test,3])
        MSE_cv = np.zeros([self.N_test,3])

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

            write_x_data.writerow(xout)
            write_y_data.writerow(yout)
            write_th_data.writerow(thout)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_x_inputs[0:,1], train_x_inputs[0:,2], zs=train_x_output, s=1)
        ax.scatter(test_x_inputs[0:,1], test_x_inputs[0:,2], c="r", zs=test_x_predic, zorder=5)
        plt.xlabel("w")
        plt.ylabel("th")
        ax.set_zlabel("dx")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_y_inputs[0:,1], train_y_inputs[0:,2], zs=train_y_output, s=1)
        ax.scatter(test_y_inputs[0:,1], test_y_inputs[0:,2], zs=test_y_predic, c="r", zorder=5)
        plt.xlabel("w")
        plt.ylabel("th")
        ax.set_zlabel("dy")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(train_th_inputs[0:,1], train_th_inputs[0:,2], zs=train_th_output, s=1)
        ax.scatter(test_th_inputs[0:,1], test_th_inputs[0:,2], zs=test_th_predic, c='r', zorder=5)
        plt.xlabel("w")
        plt.ylabel("x")
        ax.set_zlabel("dth")

        for i in range(3):
            fig = plt.figure()
            plt.plot(var[0:,i], 'b')
            plt.title("Test Variance")
            plt.xlabel("Query Index")
            plt.ylabel("Varience")

            fig = plt.figure()
            plt.plot(MSE_cv[0:,i], 'b')
            plt.title("Test Cross Validation")
            plt.xlabel("Query Index")
            plt.ylabel("MSE_cv")


    def part_b_eval(self, data_files):

        x_results = machine_learning.load_data(data_files[0], 0, 0, None)
        y_results = machine_learning.load_data(data_files[1], 0, 0, None)
        th_results = machine_learning.load_data(data_files[2], 0, 0, None)

        i = self.offset + 1

        gt_arr = np.zeros([self.N_test-2, 2])
        lwlr_arr = np.zeros([self.N_test-1, 2])

        lwlr_arr[0][0] = self.learned_data[i][6]
        lwlr_arr[0][1] = self.learned_data[i][7]

        while i < self.N_test + self.offset - 1:

            j = i - (self.offset + 1)
            gt_arr[j][0] = self.learned_data[i][6]
            gt_arr[j][1] = self.learned_data[i][7]

            dt = self.learned_data[i][5]
            dx = x_results[j][3]
            dy = y_results[j][3]

            lwlr_arr[j+1][0] = lwlr_arr[j][0] + dx * dt
            lwlr_arr[j+1][1] = lwlr_arr[j][1] + dy * dt

            i += 1

        fig = plt.figure()
        plt.plot(gt_arr[0:,0], gt_arr[0:,1])
        plt.plot(lwlr_arr[0:,0], lwlr_arr[0:,1], 'r')

def main():
    """
    Main Execution Function
    """
    part_a_2D()
    # part_a_3D()

    # data_files = ["x_data.csv", "y_data.csv", "th_data.csv"]
    #
    # go = HW2(200, 3000) #(N_test, offset)
    # go.part_b()
    # go.part_b_eval(data_files)

    plt.show()

main()
