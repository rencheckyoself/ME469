"""Calculates and Prints the results for part A6"""

#import matplotlib.pyplot as plt
import numpy as np
import robot_library


def main():
    """Main Routine"""

    #create Robot Object
    bot = robot_library.Robot([0, 0, 0])

    #Import Part A6 Data Set
    sub_num = [6, 13, 17]
    marker_set = [[1.88032539, -5.57229508], [3.07964257, 0.24942861],
                  [-1.04151642, 2.80020985]]

    robot_locs = [[2, 3, 0], [0, 3, 0], [1, -2, 0]]

    global_results = [0, 0, 0]
    error_calc = [0, 0, 0]

    for i, item in enumerate(marker_set):

        bot.new_pos = robot_locs[i]

        results = bot.read_sensor(item)

        global_results[i] = [bot.new_pos[0] + np.cos(results[1]) * results[0],
                             bot.new_pos[1] + np.sin(results[1]) * results[0]]

        error_calc[i] = [marker_set[i][0] - global_results[i][0],
                         marker_set[i][1] - global_results[i][1]]

        print "For Subject #" + str(sub_num[i]) + ":"
        print "The error in the x direction is " + str(round(error_calc[i][0]))
        print "The error in the y direction is " + str(round(error_calc[i][1]))
main()
