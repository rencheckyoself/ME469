"""Calculates and Prints the results for part A2"""

import matplotlib.pyplot as plt
import numpy as np
import robot_library


def part_2a():
    """Main Routine"""

    #create Robot Object
    bot = robot_library.Robot([0, 0, 0])

    #Part A2 Data Set
    movement_data = [[0.5, 0, 1], [0, -1/(2*np.pi), 1], [.5, 0, 1], [0, 1/(2*np.pi), 1], [.5, 0, 1]]

    # loop through data set calulate robot movement using
    # motion model
    for item in enumerate(movement_data):
        # calculate new state
        bot.make_move(item[1])

    # parse data to plot
    x_arr, y_arr, _t_arr = map(list, zip(*bot.motion_path))

    plt.plot(x_arr, y_arr, 'b')

    plt.title('Part A.2 Results')
    plt.xlabel('Robot X Position (m)')
    plt.ylabel('Robot Y Position (m)')
    plt.legend(['Robot Trajectory'])
    plt.show()

part_2a()
