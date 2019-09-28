import matplotlib.pyplot as plt
import numpy as np
import robotLibrary


def main():
    #create Robot Object
    bot = robotLibrary.Robot()

    #Import Part A2 Data Set
    movementData = [[0.5,0,1],[0,-1/(2*np.pi),1],[.5,0,1],[0,1/(2*np.pi),1],[.5,0,1]]


    for i in range(len(movementData)):

        bot.makeMove(movementData[i])

        ax.plot([bot.prevPos[0], bot.currentPos[0]], [bot.prevPos[1], bot.currentPos[1]],'b')

    #print(bot.motionPath)
    plt.axis([-.5,2,-.5,.5])

    plt.title('Part A.2 Results')
    plt.xlabel('Robot X Position (m)')
    plt.ylabel('Robot Y Position (m)')
    plt.show()


fig, ax = plt.subplots()
main()
