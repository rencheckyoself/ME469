"""Main Run File"""

import machine_learning
import numpy as np
import matplotlib.pyplot as plt

def part_a():
    """
    Function to test the performance of a learning algoithm using a sine wave
    """
    noise_var = .1
    N = 500 # Number of data points

    # Generate training data for a sine wave
    training_data_x = np.array([np.linspace(0, 2*np.pi, N)]).T
    training_data_x = np.hstack([training_data_x, np.ones([training_data_x.shape[0],1])])
    training_data_y = np.zeros(N)

    for y, x, in enumerate(training_data_x):
        training_data_y[y] = np.sin(x[0]) + np.random.normal(0,noise_var)

    # Generate test data set
    test_data_x = (2*np.pi) * np.random.rand(N,1)
    test_data_x = np.hstack([test_data_x, np.ones([test_data_x.shape[0],1])])
    test_data_pred_w = np.zeros([N,1])
    test_data_pred_uw = np.zeros([N,1])
    # Initialize LWLR Object
    ML = machine_learning.LWLR(training_data_x, training_data_y)

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
    plt.show()

def main():
    """
    Main Execution Function
    """
    part_a()

main()
