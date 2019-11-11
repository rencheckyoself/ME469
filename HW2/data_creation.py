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
    """
    Class to create training data set for ML task
    """
    def __init__(self):
        self.odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
        self.groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

        self.fin_arr = np.zeros([len(self.odom_data),7])

        self.matched_data = open('matched_data.csv', 'wb')
        self.calced_data = open('learned_data.csv', 'wb')
        self.write_matched_data = csv.writer(self.matched_data, delimiter=" ")
        self.write_calced_data = csv.writer(self.calced_data, delimiter=" ")

        # threshold for acceptable time difference
        self.time_proxitiy = 0.05

    def match_data(self):
        """
        Matches command data point to groundtruth data points and compiles
        info into one matrix
        """
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

        self.clean_for_dt()

    def clean_for_dt(self):
        """
        Only keep points with a groundtruth pose within .05 of the control.
        """
        buf = np.copy(self.fin_arr)
        # time_comp = []
        i = 0

        while i < len(buf):

            time_diff = buf[i][3] - buf[i][0]

            if time_diff > self.time_proxitiy:
                buf = np.delete(buf, i, 0)
            else:
                # time_comp.append(time_diff)
                i += 1

        # plt.plot(time_comp, 'bo')
        # plt.show()
        self.fin_arr = np.copy(buf)

    def calc_state_change(self):
        """

        Assumptions:
            The groundtruth state was recorded at the time the control was sent.

        """


        ext = np.zeros([len(self.fin_arr),5])
        i = 0

        while i < len(ext)-1:

            ext[i][0] = self.fin_arr[i][1]
            ext[i][1] = self.fin_arr[i][2]
            ext[i][2] = (self.fin_arr[i+1][4] - self.fin_arr[i][4]) / (self.fin_arr[i+1][0] - self.fin_arr[i][0])
            ext[i][3] = (self.fin_arr[i+1][5] - self.fin_arr[i][5]) / (self.fin_arr[i+1][0] - self.fin_arr[i][0])
            ext[i][4] = (self.fin_arr[i+1][6] - self.fin_arr[i][6])

            if ext[i][4] >= np.pi:
                ext[i][4] -= np.pi
            elif ext[i][4] <= -np.pi:
                ext[i][4] += np.pi

            ext[i][4] = (self.fin_arr[i+1][6] - self.fin_arr[i][6]) / (self.fin_arr[i+1][0] - self.fin_arr[i][0])

            self.write_calced_data.writerow(ext[i])

            i += 1

        w = ext[0:,1]
        v = ext[0:,0]
        x = ext[0:,2]
        y = ext[0:,3]
        th = ext[0:,4]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=x)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('x-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, th, zs=x)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('x-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=y)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('y-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, self.fin_arr[0:,6], zs=y)
        plt.xlabel('v')
        plt.ylabel('th')
        plt.title('y-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=th)
        plt.xlabel('v')
        plt.ylabel('w')
        plt.title('th-pos')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v,x, 'bo')
        plt.xlabel('v')
        plt.ylabel('x')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w,x, 'bo')
        plt.xlabel('w')
        plt.ylabel('x')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v,y, 'bo')
        plt.xlabel('v')
        plt.ylabel('y')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w,y, 'bo')
        plt.xlabel('w')
        plt.ylabel('y')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v,th, 'bo')
        plt.xlabel('v')
        plt.ylabel('th')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w,th,'bo')
        plt.xlabel('w')
        plt.ylabel('th')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,self.fin_arr[0:,6], 'bo')
        plt.xlabel('x')
        plt.ylabel('th')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y,self.fin_arr[0:,6], 'bo')
        plt.xlabel('y')
        plt.ylabel('th')

        plt.show()

    def print_csv(self):
        for _ig, row in enumerate(self.fin_arr):
            self.write_matched_data.writerow(row)

    def get_nearest_index(self, time_value):
        """
        Binary search function to match data points
        """
        lower = 0
        upper = len(self.groundtruth_data)-1

        while (upper - lower > 1):
            mid = (upper + lower) >> 1 # get the midpoint rounded down

            if (time_value >= self.groundtruth_data[mid][0]):
                lower = mid
            else:
                upper = mid

        return upper

def main():
    a = DataCreation()
    a.match_data()
    a.calc_state_change()

main()
