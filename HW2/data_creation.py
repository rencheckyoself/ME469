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

        self.fin_arr = np.zeros([len(self.odom_data), 7])

        self.calced_data = open('learning_dataset.csv', 'wb')
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


        ext = np.zeros([len(self.fin_arr), 9])
        header = ["v", "w", "dx", "dy", "dz", "dt", "x", "y", "th"]
        self.write_calced_data.writerow(header)

        i = 0

        while i < len(ext)-1:

            dt = (self.fin_arr[i+1][0] - self.fin_arr[i][0])

            ext[i][0] = self.fin_arr[i][1] #v
            ext[i][1] = self.fin_arr[i][2] #w
            ext[i][2] = (self.fin_arr[i+1][4] - self.fin_arr[i][4]) / dt #dx
            ext[i][3] = (self.fin_arr[i+1][5] - self.fin_arr[i][5]) / dt #dy
            ext[i][4] = (self.fin_arr[i+1][6] - self.fin_arr[i][6]) #dth
            ext[i][5] = dt
            ext[i][6] = self.fin_arr[i][4] #x
            ext[i][7] = self.fin_arr[i][5] #y
            ext[i][8] = self.fin_arr[i][6] #th

            if ext[i][4] >= np.pi:
                ext[i][4] -= np.pi
            elif ext[i][4] <= -np.pi:
                ext[i][4] += np.pi

            ext[i][4] = ext[i][4] / dt #dth

            # account for removed data
            if dt <= .5:
                self.write_calced_data.writerow(ext[i])
            else:
                ext[i][0] = 0
                ext[i][1] = 0
                ext[i][2] = 0
                ext[i][3] = 0
                ext[i][4] = 0
                ext[i][5] = 0
                ext[i][6] = 0
                ext[i][7] = 0
                ext[i][8] = 0

            i += 1

        w = ext[0:,1]
        v = ext[0:,0]
        x = ext[0:,2]
        y = ext[0:,3]
        th = ext[0:,4]
        dt = ext[0:,5]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=x, s=1)
        plt.xlabel('v')
        plt.ylabel('w')
        ax.set_zlabel('dx')
        ax.set_zlim(-.5, .5)
        plt.title('Change in x')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, self.fin_arr[0:,6], zs=x, s=1)
        plt.xlabel('v')
        plt.ylabel('th')
        ax.set_zlabel('dx')
        ax.set_zlim(-.5, .5)
        plt.title('Change in x')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=y, s=1)
        plt.xlabel('v')
        plt.ylabel('w')
        ax.set_zlabel('dy')
        ax.set_zlim(-.5, .5)
        plt.title('Change in y')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, self.fin_arr[0:,6], zs=y, s=1)
        plt.xlabel('v')
        plt.ylabel('th')
        ax.set_zlabel('dy')
        ax.set_zlim(-.5, .5)
        plt.title('Change in y')

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(v, w, zs=th, s=1)
        plt.xlabel('v')
        plt.ylabel('w')
        ax.set_zlabel('dth')
        plt.title("Change in theta")
        ax.set_zlim(-5, 5)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.fin_arr[0:,6], w, zs=th, s=1)
        plt.xlabel('th')
        plt.ylabel('w')
        ax.set_zlabel('dth')
        plt.title("Change in theta")
        ax.set_zlim(-1, 1)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(self.fin_arr[0:,5], w, zs=th)
        # plt.xlabel('y')
        # plt.ylabel('w')
        # ax.set_zlabel('dth')
        # ax.set_zlim(-5, 5)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v,x, 'bo')
        plt.xlabel('v')
        plt.ylabel('dx')
        plt.title("Linear Vel. vs. Change in x")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w,x, 'bo')
        plt.xlabel('w')
        plt.ylabel('dx')
        plt.title("Angular Vel. vs. Change in x")


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(v,y, 'bo')
        plt.xlabel('v')
        plt.ylabel('dy')
        plt.title("Linear Vel. vs. Change in y")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w,y, 'bo')
        plt.xlabel('w')
        plt.ylabel('dy')
        plt.title("Angular Vel. vs. Change in y")

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(v, th, 'bo')
        # plt.xlabel('v')
        # plt.ylabel('dth')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(w, th,'bo')
        plt.xlabel('w')
        plt.ylabel('dth')

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x,self.fin_arr[0:,6], 'bo', markersize=2)
        plt.xlabel('dx')
        plt.ylabel('th')
        plt.title("Change in x vs. Heading")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y,self.fin_arr[0:,6], 'bo', markersize=2)
        plt.xlabel('dy')
        plt.ylabel('th')
        plt.title("Change in y vs. Heading")

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
    plt.show()

main()
