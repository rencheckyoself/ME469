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

    # Generate training data for a sine wave
    training_data_x1 = np.array([np.linspace(0, 2*np.pi, N)]).T

    training_data_x = np.hstack([training_data_x1, np.ones([training_data_x1.shape[0], 1])])
    training_data_y = np.zeros(N)

    print training_data_x1.shape

    print training_data_x.shape

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0, noise_var)

    # Generate test data set
    test_data_x1 = (2*np.pi) * np.random.rand(N,1)
    test_data_x = np.hstack([test_data_x1, np.ones([test_data_x1.shape[0],1])])

    test_data_pred_w = np.zeros([N,1])
    test_data_pred_uw = np.zeros([N,1])

    # Initialize LWLR Object
    ML = machine_learning.LWLR(training_data_x, training_data_y, 0.4)

    # make predictions for each test data point
    for y, x in enumerate(test_data_x):
        test_data_pred_w[y] = ML.weighted_prediction(x)
        test_data_pred_uw[y] = ML.unweighted_prediction(x)

    plt.plot(training_data_x[0:,0], training_data_y, 'bo')
    plt.plot(test_data_x[0:,0], test_data_pred_w, 'ro')
    plt.plot(test_data_x[0:,0], test_data_pred_uw, 'go')

    plt.title("LWLR Test and Comparison")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

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

    # make predictions for each test data point
    for y, x in enumerate(test_data_x):
        test_data_pred_w[y] = ML.weighted_prediction(x)
        test_data_pred_uw[y] = ML.unweighted_prediction(x)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(training_data_x[0:,0], training_data_x[0:,1], zs=training_data_y)
    ax.scatter(test_data_x[0:,0], test_data_x[0:,1], zs=test_data_pred_w)
    ax.scatter(test_data_x[0:,0], test_data_x[0:,1], zs=test_data_pred_uw)

    plt.title("LWLR Test and Comparison")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(["Training Data", "Weighted Predictions", "Unweighted Predictions"])

def part_b():

    learned_data = machine_learning.load_data("learned_data.csv", 0, 0, None)

    x_data_file = open('x_data.csv', 'wb')
    write_x_data = csv.writer(x_data_file, delimiter=" ")

    y_data_file = open('y_data.csv', 'wb')
    write_y_data = csv.writer(y_data_file, delimiter=" ")

    th_data_file = open('th_data.csv', 'wb')
    write_th_data = csv.writer(th_data_file, delimiter=" ")

    N_test = 200
    N_train = learned_data.shape[0] - N_test

    # split up total data set for training and testing
    training_ldata = np.zeros([N_train, len(learned_data[0])])
    test_ldata = np.zeros([N_test, len(learned_data[0])])

    # convert to arrays and split data
    offset = 5000
    for i, row in enumerate(learned_data):
        for j, col in enumerate(row):

            if i < offset:
                training_ldata[i][j] = col
            elif i >= offset and i < offset + N_test:
                test_ldata[i - offset][j] = col
            else:
                arr_row = i - N_test
                training_ldata[arr_row][j] = col

    # Create training data sets for LWLR
    v  = np.reshape(training_ldata[0:, 0], (N_train, 1))
    w  = np.reshape(training_ldata[0:, 1], (N_train, 1))
    x  = np.reshape(training_ldata[0:, 6], (N_train, 1))
    th = np.reshape(training_ldata[0:, 8], (N_train, 1))

    dx  = np.reshape(training_ldata[0:, 2], (N_train, 1))
    dy  = np.reshape(training_ldata[0:, 3], (N_train, 1))
    dth = np.reshape(training_ldata[0:, 4], (N_train, 1))

    # Change in x training data v, w, th, 1 -> dx
    train_x_inputs = np.hstack([v, w, th, np.ones([N_train, 1])])
    train_x_output = dx

    # Change in y training data  v, w, th, 1 -> dy
    train_y_inputs = np.hstack([v, w, th, np.ones([N_train, 1])])
    train_y_output = dy

    # Change in th training data  v, w, x, 1 -> dth
    train_th_inputs = np.hstack([v, w, x, np.ones([N_train, 1])])
    train_th_output = dth

    # Generate test data set
    v  = np.reshape(test_ldata[0:, 0], (N_test, 1))
    w  = np.reshape(test_ldata[0:, 1], (N_test, 1))
    x  = np.reshape(test_ldata[0:, 6], (N_test, 1))
    th = np.reshape(test_ldata[0:, 8], (N_test, 1))

    test_x_inputs = np.hstack([v, w, th, np.ones([N_test, 1])])
    test_x_predic = np.zeros([N_test, 1])

    test_y_inputs = np.hstack([v, w, th, np.ones([N_test, 1])])
    test_y_predic = np.zeros([N_test, 1])

    test_th_inputs = np.hstack([v, w, x, np.ones([N_test, 1])])
    test_th_predic = np.zeros([N_test, 1])

    # Initialize LWLR Object
    ML_x = machine_learning.LWLR(train_x_inputs, train_x_output, 1)
    ML_y = machine_learning.LWLR(train_y_inputs, train_y_output, 1)
    ML_th = machine_learning.LWLR(train_th_inputs, train_th_output, 1)

    # make predictions for each test data point
    for q in range(N_test):
        test_x_predic[q] = ML_x.weighted_prediction(test_x_inputs[q])
        test_y_predic[q] = ML_y.weighted_prediction(test_y_inputs[q])
        test_th_predic[q] = ML_th.weighted_prediction(test_th_inputs[q])

        if q % 200 == 0:
            print q

    for p in range(N_test):

        xout = [test_x_inputs[p][0], test_x_inputs[p][1], test_x_inputs[p][2], test_x_predic[p][0]
        yout = [test_y_inputs[p][0], test_y_inputs[p][1], test_y_inputs[p][2], test_y_predic[p][0]]
        thout = [test_th_inputs[p][0], test_th_inputs[p][1], test_th_inputs[p][2], test_th_predic[p][0]]

        write_x_data.writerow(xout)
        write_y_data.writerow(yout)
        write_th_data.writerow(thout)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_x_inputs[0:,0], train_x_inputs[0:,1], zs=train_x_output)
    ax.scatter(test_x_inputs[0:,1], test_x_inputs[0:,2], zs=test_x_predic)
    plt.xlabel("w")
    plt.ylabel("th")
    ax.set_zlabel("dx")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_y_inputs[0:,0], train_y_inputs[0:,1], zs=train_y_output)
    ax.scatter(test_y_inputs[0:,1], test_y_inputs[0:,2], zs=test_y_predic)
    plt.xlabel("w")
    plt.ylabel("th")
    ax.set_zlabel("dy")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(train_th_inputs[0:,0], train_th_inputs[0:,1], zs=train_th_output)
    ax.scatter(test_th_inputs[0:,1], test_th_inputs[0:,2], zs=test_th_predic)
    plt.xlabel("w")
    plt.ylabel("x")
    ax.set_zlabel("dth")

def main():
    """
    Main Execution Function
    """
    # part_a_2D()
    # part_a_3D()

    part_b()
    plt.show()

main()
