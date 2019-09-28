import numpy as np

class Robot:
    """"A Class to interact with the motion model of a robot moving in 2D space"""

    def __init__(self):

    #   all of these positions are relative to the robot's domain
        initialPos = [0,0,0]
        self.motionPath = [[0,0,0]]
        self.currentPos = [0,0,0]
        self.prevPos = [0,0,0]

    # Function to import data from a file
    # Files must be in same directory as the python file.
    def LoadData(self, fileName, header, footer, cols):

        data = np.genfromtxt(fileName,skip_header = header, skip_footer = footer, names=True, dtype=None, delimiter=' ', usecols=cols)

        return data


    # Function to return the next point of motion over some change in time given the starting point and velocities
    #
    # Motion model consists of standard kinematics equations:
    #   new_position = initial_position + velocity * deltaT
    #
    # Input Variable Syntax:
    #        startPos = [x,y,heading] relative to the robot's domain
    #        movementSet = [v,w,dt]
    #
    # Output Variable:
    #        newPos = [x,y,heading] relative to the robot's domain
    #
    #        The new position of the robot after some change in time

    def makeMove(self, movementSet):

        vel = [0,0,0]

        #Calculate directional velocities
        self.prevPos = self.currentPos[:]
        vel[0] = movementSet[0]*np.cos(self.prevPos[2])
        vel[1] = movementSet[0]*np.sin(self.prevPos[2])
        vel[2] = movementSet[1]

        #calculate the new
        for i in range(len(self.prevPos)):

            self.currentPos[i] = self.prevPos[i] + vel[i]*movementSet[2]

        self.motionPath.append(self.currentPos[:])


    def plotPath(self):
        pass
