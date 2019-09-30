"""Calculates and Prints the results for part A3"""
import matplotlib.pyplot as plt
import numpy as np
import robot_library



# Function to import data from a file
# Files must be in same directory as the python file.

def load_data(file_name, header, footer, cols):
    """Loads in data from a text file"""

    data = np.genfromtxt(file_name, skip_header=header, skip_footer=footer,
                         names=True, dtype=None, delimiter=' ', usecols=cols)

    return data

def plot_map(landmark_data):
    """Plots the Landmarks and walls"""

    _ignore, land_x, land_y, _ignore, _ignore = map(list, zip(*landmark_data))

    plt.plot(land_x, land_y, 'ro', markersize=3)

    for _a, item in enumerate(landmark_data):
        plt.annotate('%s' %item[0], xy=(item[1], item[2]), xytext=(3, 3),
                     textcoords='offset points')

    walls_x = [land_x[1], land_x[4], land_x[9], land_x[11], land_x[12], land_x[13], land_x[14],
               land_x[6], land_x[5], land_x[2], land_x[0], land_x[3], land_x[4]]
    walls_y = [land_y[1], land_y[4], land_y[9], land_y[11], land_y[12], land_y[13], land_y[14],
               land_y[6], land_y[5], land_y[2], land_y[0], land_y[3], land_y[4]]

    plt.plot(walls_x, walls_y, 'k')

    walls_x = [land_x[10], land_x[8], land_x[7]]
    walls_y = [land_y[10], land_y[8], land_y[7]]

    plt.plot(walls_x, walls_y, 'k', label='_nolegend_')

def main():
    """Main Routine"""


# load in data files

    odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
    meas_data = load_data('ds1_Measurement.dat', 3, 0, [0, 4, 6, 7])
    marker_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])
    bar_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
    groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

# create Robot Object

    bot = robot_library.Robot([groundtruth_data[0][1], groundtruth_data[0][2],
                               groundtruth_data[0][3]])

# Transform Measurement from Barcode to Subject
    for _counter, item in enumerate(meas_data):
        for _match, match in enumerate(bar_data):
            if match[1] == item[1]:
                item[1] = match[0]
                break

# plot Landmarks and Walls

    plot_map(marker_data)

# Plot Groundtruth Data

    _ignore, ground_x, ground_y, _ground_t = map(list, zip(*groundtruth_data))


    #ground_x = ground_x[0:20000]
    #ground_y = ground_y[0:20000]

    plt.plot(ground_x[0], ground_y[0], 'kd', markersize=3, label='_nolegend_')
    plt.plot(ground_x, ground_y, 'g')
    plt.plot(ground_x[-1], ground_y[-1], 'ko', markersize=3, label='_nolegend_')

# Plot Odometry Dataset

    for i, cur_action in enumerate(odom_data):

        if i + 1 >= len(odom_data):
            break

        movement_data = [odom_data[i][1], odom_data[i][2], odom_data[i+1][0] - cur_action[0]]
        bot.make_move(movement_data)


    print bot.motion_path[-20:-1]

    x_arr, y_arr, _t_arr = map(list, zip(*bot.motion_path))

    plt.plot(x_arr, y_arr, 'b')

    plt.legend(['Landmark', 'Wall', 'Groundtruth', 'Robot Trajectory'])
    plt.xlabel('World X Position (m)')
    plt.ylabel('World Y Position (m)')
    plt.title('Odometry Data Vs. Groundtruth Data')

    plt.autoscale(True)
    plt.show()

main()
