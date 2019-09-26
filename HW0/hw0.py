import numpy as np
import math
import matplotlib.pyplot as plt


def LoadData(fileName, header, footer, cols):
    data = np.genfromtxt(fileName,skip_header = header, skip_footer = footer, names=True, dtype=None, delimiter=' ', usecols=cols)

    return data

#load in data files
odomData = LoadData('ds1_Odometry.dat',3,1, [0,4,5])
measData = LoadData('ds1_Measurement.dat',3,1,[0,4,6,7])
markerData = LoadData('ds1_Landmark_Groundtruth.dat',3,1,[0,2,4,6,8])

# First establish starting position
def makeMove(startPos,movementSet):

    #startPos = [x,y,heading]
    #movementSet = [v,w,dt]
    newPos = [0,0,0]
    vel = [0,0,0]
    vel[0] = movementSet[0]*math.cos(startPos[2])
    vel[1] = movementSet[0]*math.sin(startPos[2])
    vel[2] = movementSet[1]

    for i in range(len(startPos)):

        newPos[i] = startPos[i] + vel[i]*movementSet[2]

    return newPos

intialPos = [0,0,0] #[x,y,theta]
movementData = LoadData('partA2.dat',0,0,[0,1,2])

fig, ax = plt.subplots()

print(movementData)

curPos = intialPos

for i in range(len(movementData)):
    nextPos = makeMove(curPos,movementData[i])

    ax.plot([curPos[0], nextPos[0]],[curPos[1], nextPos[1]])

    curPos = nextPos

plt.axis([-1.5,1.5,-1.5,1.5])
plt.show()
