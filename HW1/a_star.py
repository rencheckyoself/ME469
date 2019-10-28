""" Library to run an AStar search algoritim and simulate a robot driving the path"""
from __future__ import division
import heapq
# from pprint import pprint
import numpy as np
# import time
import matplotlib.pyplot as plt

# turn on interactive plotting
plt.ion()

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

def plot_show():
    """
    Function to turn off interactive mode and call the show function so that
    plots remain visible until user closes them
    """
    plt.ioff()
    plt.show()

class Node(object):
    """
    Object to store data of an individual node created by the A* Algorithm

    INPUT:
        next_id: A unique ID number for the new node being created
        par_id: The unique ID of the parent node this new node is a neighbor of
        x_pos: The x position in world coordinates of the new node
        y_pos: The y position in world coordinates of the new node
    """
    def __init__(self, next_id, par_id, x_pos, y_pos):

        self.f_val = 0 #f
        self.h_val = 0 #h
        self.g_val = 0 #g
        self.node_id = next_id #node_id
        self.parent_id = par_id #parent_id
        self.x_pos_w = x_pos # world x_pos
        self.y_pos_w = y_pos #world y_pos
        self.x_pos_g = 0 #grid x_pos
        self.y_pos_g = 0 #grid y_pos

        self.sort_val = (self.f_val, self.h_val) #sort value to optimize heap

    # Function used by heapq for proper sorting, sort is based on lowest f value
    # then by lowest h value
    def __lt__(self, other):
        return self.sort_val < other.sort_val

class AStar(object):
    """
    Obeject for A* algoritim logic

    INPUTS:
        start_coord: x and y position of a starting location
        goal_coord: x and y position of a goal location
        cell_size: grid cell height and width
        inflate: radius to add to each landmark location

    OPTIONAL INPUTS:
        online_check: Determines if the search as apriori knowledge of the map.
                      Default is True(1), meaning the search does not have apriori
                      knowledge.
        robot_check: Determines if the search is fully completed prior to running
                     a robot simulation.
                     Default is False(0), meaing full search is completed before
                     running robot simulation
        grid_obj: Pass an existing Grid() object. Default is None
    """

    def __init__(self, start_coord, goal_coord, cell_size, inflate, online_check=1,
                 robot_check=0, grid_obj=None):

        # Create Grid if not alread passed from Robot Object
        if robot_check == 0:
            self.field = Grid(cell_size, inflate)
        else:
            self.field = grid_obj

        self.online = online_check

        self.goal_loc_w = goal_coord
        self.goal_loc_g = self.field.get_grid_index(self.goal_loc_w[0], self.goal_loc_w[1])
        self.goal_node = None

        self.fin_path = [None]

        self.start_node = Node(0, None, start_coord[0], start_coord[1])
        self.start_node.x_pos_g, self.start_node.y_pos_g = self.field.get_grid_index(
            self.start_node.x_pos_w, self.start_node.y_pos_w)

        self.next_id = 1

        self.open_list = [self.start_node]
        self.closed_list = []

        heapq.heapify(self.open_list)

    def start_planning(self):
        """
        The main routine to call to complete the full A* search
        """

        print ("Planning for path from " + str([self.start_node.x_pos_w, self.start_node.y_pos_w])
               + " to " + str(self.goal_loc_w))

        result = self.find_path()

        # Plot all nodes function
        self.plot_all_nodes()

        # Plot final path function
        if result == 1:

            plt.title("Path From " + str([self.start_node.x_pos_w, self.start_node.y_pos_w]) +
                      " To " + str(self.goal_loc_w))
            self.plot_final_path()

        else:
            plt.title("Did not find the goal")

    def find_path(self):
        """
        Function to loop the open list until the goal is found
        """

        found_end = 0
        while found_end == 0:

            # Check if the open list is empty, if so send code to nofity that the
            # goal was not found
            if len(self.open_list) == 0:
                found_end = 2

            else:
                eval_node = heapq.heappop(self.open_list)

            # Check if current node is the goal, if so send code to notify that
            # the goal was found
            if self.check_for_goal(eval_node):
                found_end = 1
                self.goal_node = eval_node
                self.closed_list.append(eval_node)

            # Otherwise add the current node to the closed list and evaluate
            # its 8 neighbors
            else:
                self.closed_list.append(eval_node)
                self.analyze_neighbors(eval_node)

        return found_end

    def analyze_neighbors(self, cur_node):
        """
        Evalute all the neighbors of a given node

        INPUT:
            cur_node: The current Node() that the search has identified as a node
                      on the lowest cost path
        """
        x_search = [-self.field.cell_size, 0, self.field.cell_size]
        y_search = x_search[:]

        # Loop through all 8 potential neighbor nodes
        for _ig, x in enumerate(x_search):
            for _ig, y in enumerate(y_search):

                pot_node_x_w = round(cur_node.x_pos_w + x, 2)
                pot_node_y_w = round(cur_node.y_pos_w + y, 2)

                # Skip over the 0,0 combination
                if x == 0 and y == 0:
                    continue

                # Check if potential node location is out of bounds, if so skip over
                elif (pot_node_x_w < self.field.x_axis[0] or
                      pot_node_x_w > self.field.x_axis[1] or
                      pot_node_y_w < self.field.y_axis[0] or
                      pot_node_y_w > self.field.y_axis[1]):
                    continue

                # Check if the potential node is on the closed list, if so skip over
                else:
                    on_cl = 0
                    for _ig, closed_node in enumerate(self.closed_list):

                        if (pot_node_x_w == closed_node.x_pos_w and
                                pot_node_y_w == closed_node.y_pos_w):
                            on_cl = 1
                            break

                if on_cl == 1:
                    continue

                # If the function has reached this point, it is confirmed that
                # the potential node is valid for evaluation.

                # Check if neighbor node is on the open list, if so compare the
                # g value and reevalutate if nessissary.
                match_found = 0
                for _ig, open_node in enumerate(self.open_list):

                    # check for match
                    if (pot_node_x_w == open_node.x_pos_w and
                            pot_node_y_w == open_node.y_pos_w):
                        match_found = 1

                        temp_g = self.calc_g(open_node, cur_node, x, y)

                        # Compare g values, and update the node if the statement
                        # is true.
                        if temp_g < open_node.g_val:
                            open_node.g_val = temp_g
                            open_node.f_val = open_node.g_val + open_node.h_val
                            open_node.sort_val = (open_node.f_val, open_node.h_val)
                            open_node.parent_id = cur_node.node_id

                        if match_found == 1:
                            break

                # If the previous loop did not find a match on the open list,
                # then create a new Node object and add it to the open list
                if match_found == 0:

                    # create a new node
                    nei_node = Node(self.next_id, cur_node.node_id, pot_node_x_w, pot_node_y_w)

                    nei_node.x_pos_g, nei_node.y_pos_g = self.field.get_grid_index(nei_node.x_pos_w,
                                                                                   nei_node.y_pos_w)
                    nei_node.g_val = self.calc_g(nei_node, cur_node, x, y)
                    nei_node.h_val = self.calc_h(nei_node)
                    nei_node.f_val = round(nei_node.g_val + nei_node.h_val, 2)
                    nei_node.sort_val = (nei_node.f_val, nei_node.h_val)

                    self.next_id += 1

                    heapq.heappush(self.open_list, nei_node)

        # If the search algorithm is run in "online" mode, the open list is
        # restricted to only the valid neighbor nodes.
        #
        # Once all the valid neighbors have been identified the best option is
        # popped of the heap, the open_list is reset, and then the best option
        # is readded to the heap for compatibility with self.find_path()
        if self.online == 1:
            best_nei = heapq.heappop(self.open_list)
            self.open_list = []
            heapq.heappush(self.open_list, best_nei)

    def calc_g(self, node_new, node_cur, shift_x, shift_y):
        """
        The true cost function for the A* algorithm

        INPUT:
            node_new: a valid neighbor node of the current node
            node_cur: the current node identified as on the lowest cost path
            shift_x: the x position of the neighbor relative to the current node
            shift_y: the y position of the neighbor relative to the current node

        OUTPUT:
            g_new: the true cost to reach the neighbor node given the current node
        """

        # Check that the neighbor node is not occupied by a landmark
        if self.field.cell_cost[node_new.x_pos_g, node_new.y_pos_g] != 1000:
            g_new = node_cur.g_val + ((shift_x / self.field.cell_size)**2 +
                                      (shift_y / self.field.cell_size)**2)**.5

        # If occupied, then add 1000 to the true cost
        else:
            g_new = node_cur.g_val + 1000

        return g_new

    def calc_h(self, node_new):
        """
        The hueristic function for the A* algorithm

        INPUT:
            node_new: a valid neighbor node of the current node

        OUTPUT:
            h_new: the estimated cost to reach the goal from the node provided
        """

        x_diff = abs(self.goal_loc_g[0] - node_new.x_pos_g)
        y_diff = abs(self.goal_loc_g[1] - node_new.y_pos_g)

        # Diagonal - Chebyshev Distance
        h_new = (x_diff + y_diff) + (1 - 2) * min(x_diff, y_diff)

        # Euclidean - Shortest Path
        # h_new = (x_diff**2 + y_diff**2)**.5

        return h_new

    def check_for_goal(self, node):
        """
        Check if the node provided mathces the goal location

        INPUT:
            node: any node object
        """

        if(self.goal_loc_w[0] == node.x_pos_w and
           self.goal_loc_w[1] == node.y_pos_w):
            check = True
        else:
            check = False

        return check

    def plot_all_nodes(self):
        """
        Function to plot all nodes on the open list and closed list. The results
        will appear as a black dot on the plot.

        This allows visualization for all nodes that the algorithm has evaluated
        and kept track of.
        """
        x_arr = np.zeros(len(self.closed_list) + len(self.open_list) - 1)
        y_arr = np.zeros(len(self.closed_list) + len(self.open_list) - 1)
        index = 0

        # Assemble arrays of x and y coords
        while index < len(self.closed_list) + len(self.open_list) - 1:
            for cl_i, node in enumerate(self.closed_list):
                x_arr[index] = node.x_pos_w
                y_arr[index] = node.y_pos_w
                index = cl_i

            for ol_i, node in enumerate(self.open_list):
                x_arr[index] = node.x_pos_w
                y_arr[index] = node.y_pos_w
                index = ol_i + len(self.closed_list)

        # Plot and update the figure
        plt.plot(x_arr, y_arr, 'ko', markersize=1)
        self.field.fig.canvas.draw()

    def plot_final_path(self):
        """
        Function to plot all nodes that make up the identified lowest cost path.
        The results will appear as a red line with markers.

        Also compiles the final path in order from start to finish. The variable
        is used to pass the path to a Robot() object.

        This allows visualization of the final path found by the algoithm
        """

        x_arr = [self.goal_node.x_pos_w]
        y_arr = [self.goal_node.y_pos_w]

        next_par_id = self.goal_node.parent_id

        self.fin_path = [self.goal_node]

        # Assemble arrays of x and y coordinates of all the nodes on the final
        # path.
        while next_par_id is not None:
            for _ig, node in enumerate(self.closed_list):
                if node.node_id == next_par_id:
                    x_arr.append(node.x_pos_w)
                    y_arr.append(node.y_pos_w)
                    next_par_id = node.parent_id
                    self.fin_path.append(node)
                    continue

        # Reorder the final path from start to goal
        self.fin_path.reverse()

        # Plot and Update
        plt.plot(x_arr, y_arr, 'ro-', markersize=2, linewidth=2)
        self.field.fig.canvas.draw()

class Grid(object):
    """
    Object that controls the grid for a given search or simulation

    INPUT:
        width: the distance of the cell width (and height since the cell is a square)
        addition: value to add/subtract from the land mark location to apply a buffer radius
    """

    def __init__(self, width, addition):

        self.cell_size = width

        # Set overall boundary
        self.x_axis = [-2, 5]
        self.y_axis = [-6, 6]

        # Read in landmark data
        self.marker_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])

        # Initialize the costs associated with each node
        self.x_lim = np.arange(self.x_axis[0], self.x_axis[1], self.cell_size)
        self.y_lim = np.arange(self.y_axis[0], self.y_axis[1], self.cell_size)
        self.cell_cost = np.ones([len(self.x_lim), len(self.y_lim)])

        # compare landmark location to x and y values
        for _x, marker in enumerate(self.marker_data):

            x_hit_min, y_hit_min = self.get_grid_index(marker[1] - addition, marker[2] - addition)
            x_hit_max, y_hit_max = self.get_grid_index(marker[1] + addition, marker[2] + addition)

            x_hit = np.arange(x_hit_min, x_hit_max, 1)
            y_hit = np.arange(y_hit_min, y_hit_max, 1)

            # Resructure variable for the case that x_hit_min = x_hit_max
            if len(x_hit) == 0:
                x_hit = [x_hit_min]

            # Resructure variable for the case that y_hit_min = y_hit_max
            if len(y_hit) == 0:
                y_hit = [y_hit_min]

            # Adjust value(s) in the cost map for the obstacle
            for _ig, mark_x in enumerate(x_hit):
                for _ig, mark_y in enumerate(y_hit):
                    self.cell_cost[mark_x][mark_y] = 1000

        # Parse landmark data to plot
        _ignore, land_x, land_y, _ignore, _ignore = map(list, zip(*self.marker_data))

        self.fig, self.ax = plt.subplots()

        plt.plot(land_x, land_y, 'c^', markersize=3)

        plt.xticks(np.arange(self.x_axis[0], self.x_axis[1]+self.cell_size, 1))
        # plt.xticks(np.arange(self.x_axis[0], self.x_axis[1]+self.cell_size, .1))
        plt.yticks(np.arange(self.y_axis[0], self.y_axis[1]+self.cell_size, 1))
        # plt.yticks(np.arange(self.y_axis[0], self.y_axis[1]+self.cell_size, .1))

        plt.imshow(self.cell_cost.T, cmap="Purples", origin='lower', extent=[-2, 5, -6, 6])

        plt.grid(which="major", color='k', linewidth=1)
        # plt.grid(which="minor", color='-k')

        plt.ylabel("World Y Position (m)")
        plt.xlabel("World X Position (m)")
        plt.axis([-2, 5, -6, 6])

        # show Figure
        self.fig.canvas.draw()

    def get_grid_index(self, data_set_x, data_set_y):
        """
        Fucntion to convert from world coordinates to grid coordinates.

        INPUT:
            data_set_x, data_set_y: world position x and y value

        OUTPUT:
            grid_x, grid_y: grid coordinates of the given world coordinates

        """

        for x, graph_x in enumerate(self.x_lim):
            if (data_set_x >= graph_x and
                    data_set_x <= graph_x + self.cell_size):
                grid_x = x

        for y, graph_y in enumerate(self.y_lim):
            if (data_set_y >= graph_y and
                    data_set_y <= graph_y + self.cell_size):
                grid_y = y

        return grid_x, grid_y

class Robot(object):
    """
    Object to control the robot simulation

    INPUTS:
        start_coord: x and y position of a starting location
        goal_coord: x and y position of a goal location

    OPTIONAL INPUTS:
        final_path: a list of Node() objects that desribe the path from start to goal.
                    If no path is provided, variables will be initialized to run an A*
                    search as the robot traverses the map. Default is None.
        grid_obj: Pass an existing Grid() object. Default is None.
        cell_size: grid cell height and width. Default is 0.1
        inflate: radius to add to each landmark location. Default is 0.3
    """

    def __init__(self, start_loc, goal_loc, final_path=None, grid_obj=None,
                 cell_size=.1, inflate=.3):

        print "Driving path from " + str(start_loc) + " to " + str(goal_loc)

        # Create grid for simulatation if one wasn't provided
        if grid_obj is None:
            self.grid = Grid(cell_size, inflate)
            plt.title("Path From " + str(start_loc) + " To " + str(goal_loc))
            self.grid.fig.canvas.draw()
        else:
            self.grid = grid_obj

        # Initialize variables depending if the A* search was already completed.
        if final_path is None:
            self.cur_state = [start_loc[0], start_loc[1], (3*np.pi)/2]
            self.target_state = self.cur_state[:]
            self.path = []
            self.tracking = -1
            self.goal_state = goal_loc[:]

            # create the AStar object used by the Robot Simulation
            self.astar = AStar(start_loc, goal_loc, cell_size, inflate,
                               robot_check=1, grid_obj=self.grid)
            self.robot_target_node = heapq.heappop(self.astar.open_list)
            self.astar.closed_list.append(self.robot_target_node)
        else:
            self.cur_state = [final_path[0].x_pos_w, final_path[0].y_pos_w, (3*np.pi)/2]
            self.target_state = self.cur_state[:]
            self.goal_state = [final_path[-1].x_pos_w, final_path[-1].y_pos_w]
            self.path = final_path
            self.tracking = 0

        self.vel = [0, 0] #velocity vector [v,w]
        self.robot_path = [self.cur_state[:]] #final path of robot

        self.acc_lim = [0.288, 5.579] # m/s^2, rad/sec^2

        self.th_thresh = .08 #radians
        self.d_thresh = .025 #meters

        self.k_th = 3 #proportional gain for turning
        self.k_d = .1 #proportional gain for moving forward

        self.dt = 0.1 #timestep

        self.go_to_goal()

    def find_target(self):
        """
        Function to identify the next target node for the robot to travel to.

        If the path was provided, it sets the target to the next node on the path

        If no path was provided, it uses the online A* search to identify the
        lowest cost neighbor and sets it as the new target.
        """

        # Logic if the path was not provided
        if self.tracking == -1:

            #check if current target the robot has just arrived at is the goal
            if self.astar.check_for_goal(self.robot_target_node) == 1:
                arrived = 1

            else:

                # analyze neighbor nodes and add the lowest cost to open list
                self.astar.analyze_neighbors(self.robot_target_node)

                # retrive result form neighbor analysis and set target state
                self.robot_target_node = heapq.heappop(self.astar.open_list)
                self.astar.closed_list.append(self.robot_target_node)
                self.target_state = [self.robot_target_node.x_pos_w,
                                     self.robot_target_node.y_pos_w]

                self.grid.ax.plot(self.robot_target_node.x_pos_w,
                                  self.robot_target_node.y_pos_w,
                                  'ro', markersize=1, zorder=6)
                arrived = 0

        # Logic if the path was provided
        else:
            self.tracking += 1

            if self.tracking >= len(self.path):
                arrived = 1

            else:
                arrived = 0
                self.target_state = [self.path[self.tracking].x_pos_w,
                                     self.path[self.tracking].y_pos_w]
        return arrived

    def go_to_goal(self):
        """
        Function to simulate robot movement by calculating a linear and angular
        velocity from a proportional control. The command has noise included to
        model a real system. There is also a constrain for an acceleration limit.

        The simulation assumes the robot is allowed to know its true position, so
        the new position is calculated based the calculated velocity command.
        """

        goal_check = 0
        while goal_check == 0:

            prev_vel = [self.vel[0], self.vel[1]]

            x_dist = self.target_state[0] - self.cur_state[0]
            y_dist = self.target_state[1] - self.cur_state[1]

            # determine the distance between the robot and the target
            delta_th = round(np.arctan2(y_dist, x_dist) - self.cur_state[2], 3)
            delta_d = round((x_dist**2 + y_dist**2)**.5, 3)

            # determine if the robot is at the target
            if delta_d <= self.d_thresh:

                goal_check = self.find_target()

                x_dist = self.target_state[0] - self.cur_state[0]
                y_dist = self.target_state[1] - self.cur_state[1]

                delta_th = round(np.arctan2(y_dist, x_dist) - self.cur_state[2], 1)
                delta_d = round((x_dist**2 + y_dist**2)**.5, 3)

            else:

                #calculate the linear veloctiy command with added noise
                eps = np.random.normal(0, .001)
                self.vel[0] = self.k_d * delta_d + eps

                # adjust command for the linear acceleration limit
                if (self.vel[0] - prev_vel[0]) / self.dt > self.acc_lim[0]:
                    self.vel[0] = prev_vel[0] + self.dt * self.acc_lim[0]

                elif (self.vel[0] - prev_vel[0]) / self.dt < -self.acc_lim[0]:
                    self.vel[0] = prev_vel[0] - self.dt * self.acc_lim[0]

                #remap difference in heading to turn in the shortest direction
                if delta_th >= (7*np.pi/6):
                    delta_th = delta_th - 2*np.pi
                elif delta_th < -(7*np.pi/6):
                    delta_th = delta_th + 2*np.pi

                #calculate the angular veloctiy command with added noise
                eps = np.random.normal(0, .0001)
                self.vel[1] = self.k_th * delta_th + eps

                # adjust command for the angular acceleration limit
                if (self.vel[1] - prev_vel[1]) / self.dt > self.acc_lim[1]:
                    self.vel[1] = prev_vel[1] + self.dt * self.acc_lim[1]

                elif (self.vel[1] - prev_vel[1]) / self.dt < -self.acc_lim[1]:
                    self.vel[1] = prev_vel[1] - self.dt * self.acc_lim[1]

                #Calculate new position based on the velocity command
                self.cur_state[0] = (self.cur_state[0] + self.vel[0] *
                                     np.cos(self.cur_state[2]) * self.dt)

                self.cur_state[1] = (self.cur_state[1] + self.vel[0] *
                                     np.sin(self.cur_state[2]) * self.dt)

                self.cur_state[2] = self.cur_state[2] + self.vel[1] * self.dt

                self.robot_path.append(self.cur_state[:])

        # plot the final robot path
        self.plot_robot_traj()

    def plot_robot_traj(self):
        """
        Plot the final robot trajectory using arrows to show the trajectory
        """

        for _ig, state in enumerate(self.robot_path):
            arrow_x = .001*np.cos(state[2])
            arrow_y = .001*np.sin(state[2])
            self.grid.ax.arrow(state[0], state[1], arrow_x, arrow_y, head_width=.1,
                               color='black', zorder=1)

        self.grid.fig.canvas.draw()
