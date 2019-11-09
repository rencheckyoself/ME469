# Clean data
import heapq as hq
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

# import data and assign each control and GT data point a unique ID
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


class DataCreation(object):

    def __init__(self):
        self.odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
        self.groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

        self.fin_arr = np.zeros([len(self.odom_data),7])

        self.output_file = open('total_data.csv', 'wb')
        self.writer = csv.writer(self.output_file)

    def match_data(self):

        time_diff = np.zeros(len(self.odom_data))
        for i, row in enumerate(self.fin_arr):

            # get next odom row
            odom_row = self.odom_data[i]

            row[0] = odom_row[0]
            row[1] = odom_row[1]
            row[2] = odom_row[2]

            # get groundtruth data point to compare to odom
            index = self.get_nearest_index(odom_row[0])
            data_row = self.groundtruth_data[index]

            row[3] = data_row[0]
            row[4] = data_row[1]
            row[5] = data_row[2]
            row[6] = data_row[3]

    def clean_for_dt(self):
        """
        Only keep points with a groundtruth pose within .25 of the control.
        """
        buf = np.copy(self.fin_arr)
        # time_comp = []
        i = 0

        while i < len(buf):

            time_diff = buf[i][3] - buf[i][0]

            if time_diff > .25:
                buf = np.delete(buf, i, 0)
            else:
                # time_comp.append(time_diff)
                i += 1

        # plt.plot(time_comp, 'bo')
        # plt.show()
        self.fin_arr = np.copy(buf)

    def calc_state_change(self):
        ext = np.zeros([len(self.fin_arr),5])
        i = 0
        min_dth = 0
        while i < len(ext)-1:

            ext[i][0] = self.fin_arr[i][1]
            ext[i][1] = self.fin_arr[i][2]
            ext[i][2] = self.fin_arr[i+1][4] - self.fin_arr[i][4]
            ext[i][3] = self.fin_arr[i+1][5] - self.fin_arr[i][5]
            ext[i][4] = self.fin_arr[i+1][6] - self.fin_arr[i][6]

            if ext[i][4] >= np.pi:
                ext[i][4] -= np.pi
            elif ext[i][4] <= -np.pi:
                ext[i][4] += np.pi

            if min_dth > ext[i][4]:
                min_dth = ext[i][4]
                check = i

            i += 1

        print max(self.fin_arr[0:,6]), min(self.fin_arr[0:,6])

        print i, min_dth

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ext[0:,0], ext[0:,1], zs=ext[0:,2])
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('x-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ext[0:,0], ext[0:,1], zs=ext[0:,3])
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('y-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(ext[0:,0], ext[0:,1], zs=ext[0:,4])
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('th-pos')

        plt.show()


    def print_csv(self):
        for _ig, row in enumerate(self.fin_arr):
            self.writer.writerow(row)

    def get_nearest_index(self, time_value):
        """
        Binary search function
        """
        n = len(self.groundtruth_data)

        lower = 0
        upper = n-1

        while (upper - lower > 1):
            mid = (upper + lower) >> 1 # get the midpoint

            if (time_value >= self.groundtruth_data[mid][0]):
                lower = mid
            else:
                upper = mid

        return upper

def main():
    a = DataCreation()
    a.match_data()
    a.clean_for_dt()
    a.print_csv()
    a.calc_state_change()





main()
# Check that each control has a unique GT pose ID

# Calculate the difference in pose information from the previous command

# plot v x w x x
# plot v x w x y
# plot w x th
