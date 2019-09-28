import numpy as np
import matplotlib.pyplot as plt
import robotLibrary


def main():
    #create Robot Object
    bot = Robot()

    #load in data files
    odomData = bot.LoadData('ds1_Odometry.dat', 3, 0, [0,4,5])
    measData = bot.LoadData('ds1_Measurement.dat', 3, 0,[0,4,6,7])
    markerData = bot.LoadData('ds1_Landmark_Groundtruth.dat', 3, 0, [0,2,4,6,8])
    barData = bot.LoadData('ds1_Barcodes.dat', 3, 0, [0,3])
    groundtruthData = bot.LoadData('ds1_Groundtruth.dat', 3, 0, [0,3,5,7])

    #Import Part A2 Data Set
    movementData = [[0.5,0,1],[0,-1/(2*np.pi),1],[.5,0,1],[0,1/(2*np.pi),1],[.5,0,1]]


    for i in range(len(movementData)):

        bot.makeMove(movementData[i])
        ax.plot([bot.prevPos[0], bot.currentPos[0]], [bot.prevPos[1], bot.currentPos[1]],'b')

    print(bot.motionPath)
    plt.axis([-2,2,-2,2])
    plt.show()

fig, ax = plt.subplots()
main()
