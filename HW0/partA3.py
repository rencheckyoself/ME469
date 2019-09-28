import numpy as np
import matplotlib.pyplot as plt
import robotLibrary


def main():
    # create Robot Object
    bot = robotLibrary.Robot()

    # load in data files
    odomData = bot.LoadData('ds1_Odometry.dat', 3, 0, [0,4,5])
    measData = bot.LoadData('ds1_Measurement.dat', 3, 0,[0,4,6,7])
    markerData = bot.LoadData('ds1_Landmark_Groundtruth.dat', 3, 0, [0,2,4,6,8])
    barData = bot.LoadData('ds1_Barcodes.dat', 3, 0, [0,3])
    groundtruthData = bot.LoadData('ds1_Groundtruth.dat', 3, 0, [0,3,5,7])

# Plot Odometry Dataset
    for i in range(len(odomData)-1):

        movementData = [odomData[i][1], odomData[i][2], odomData[i+1][0] - odomData[i][0]]

        bot.makeMove(movementData)

    xArr, yArr , tArr= map(list, zip(*bot.motionPath))

    plt.plot(xArr, yArr, 'b')

    plt.legend([]'Robot Trajectory'])
    plt.xlabel('World X Position (m)')
    plt.xlabel('World Y Position (m)')
    plt.title('Odometry Data')

    plt.autoscale(True)
    plt.show()

main()
