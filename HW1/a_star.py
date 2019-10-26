""" hi """
from __future__ import division
import numpy as np
import heapq
import time
from pprint import pprint
import matplotlib.pyplot as plt

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
    plt.ioff()
    plt.show()

class Node(object):
    """
    Node object to store data of an individual node
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

    def __lt__(self, other):
        return self.sort_val < other.sort_val

class AStar(object):
    """
    contains logic for A* algoritim
    """
    def __init__(self, start_coord, goal_coord, cell_size, inflate, online_check):

        print "Planning for path from " + str(start_coord) + " to " + str(goal_coord)

        self.field = Grid(cell_size, inflate)

        self.online = online_check

        self.odom_data = load_data('ds1_Odometry.dat', 3, 0, [0, 4, 5])
        self.meas_data = load_data('ds1_Measurement.dat', 3, 0, [0, 4, 6, 7])
        self.bar_data = load_data('ds1_Barcodes.dat', 3, 0, [0, 3])
        self.groundtruth_data = load_data('ds1_Groundtruth.dat', 3, 0, [0, 3, 5, 7])

        self.node_master_list = np.empty((len(self.field.x_lim + 1) * len(self.field.y_lim + 1), 7))

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
        Plan the path.
        """
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
        A* Search Algorithm
        """

        found_end = 0
        counter = 0
        while found_end == 0:
            counter += 1

            # print "---------------------------------"
            # print len(self.open_list)

            if len(self.open_list) == 0:
                found_end = 2
            else:
                eval_node = heapq.heappop(self.open_list)

        #check if current node is the goal
            if self.check_for_goal(eval_node):
                found_end = 1
                self.goal_node = eval_node
                self.closed_list.append(eval_node)
            else:

                # if counter % 10 == 0:
                #     print counter
                #     print eval_node.h_val
                #     found_end = 2
                #     return found_end
                self.closed_list.append(eval_node)
                self.analyze_neighbors(eval_node)
                #
                # if test == 1:
                #     found_end = 3
        return found_end

    def analyze_neighbors(self, cur_node):
        """
        find and analyze all neighbor nodes
        """
        x_search = [-self.field.cell_size, 0, self.field.cell_size]
        y_search = x_search[:]

        # print cur_node.node_id
        #look at all 8 neighbor nodes
        for _ig, x in enumerate(x_search):
            for _ig, y in enumerate(y_search):

                pot_node_x_w = round(cur_node.x_pos_w + x, 2)
                pot_node_y_w = round(cur_node.y_pos_w + y, 2)
                # print x, y, pot_node_x_w, pot_node_y_w
                # quick skip for 0,0 shift
                if x == 0 and y == 0:

                    continue

                #check if node is out of bounds, if so skip over
                elif (pot_node_x_w < self.field.x_axis[0] or
                      pot_node_x_w > self.field.x_axis[1] or
                      pot_node_y_w < self.field.y_axis[0] or
                      pot_node_y_w > self.field.y_axis[1]):

                    continue

                else:
                # check if node is on the closed list, if so skip over
                    on_cl = 0
                    for _ig, closed_node in enumerate(self.closed_list):
                        if (pot_node_x_w == closed_node.x_pos_w and
                                pot_node_y_w == closed_node.y_pos_w):

                            on_cl = 1
                            break

                if on_cl == 1:
                    continue

                match_found = 0

                # check if neighbor is on the open list
                for _ig, open_node in enumerate(self.open_list):

                    if (pot_node_x_w == open_node.x_pos_w and
                            pot_node_y_w == open_node.y_pos_w):

                        # if a match is found then compare g values.
                        match_found = 1
                        temp_g = self.calc_g(open_node, cur_node, x, y)

                        if temp_g < open_node.g_val:
                            # print "modified node " + str(open_node.node_id)
                            open_node.g_val = temp_g
                            open_node.f_val = open_node.g_val + open_node.h_val
                            open_node.sort_val = (open_node.f_val, open_node.h_val)
                            open_node.parent_id = cur_node.node_id

                        if match_found == 1:
                            break

                if match_found == 0:

                    # create a new node
                    nei_node = Node(self.next_id, cur_node.node_id, pot_node_x_w, pot_node_y_w)

                    nei_node.x_pos_g, nei_node.y_pos_g = self.field.get_grid_index(nei_node.x_pos_w, nei_node.y_pos_w)

                    nei_node.g_val = self.calc_g(nei_node, cur_node, x, y)
                    nei_node.h_val = self.calc_h(nei_node)
                    nei_node.f_val = round(nei_node.g_val + nei_node.h_val, 2)

                    nei_node.sort_val = (nei_node.f_val, nei_node.h_val)

                    self.next_id += 1

                    # pprint(vars(nei_node))

                    # if self.check_for_goal(nei_node):
                    #     goal_created = 1

                    heapq.heappush(self.open_list, nei_node)

        if self.online == 1:
            best_nei = heapq.heappop(self.open_list)
            self.open_list = []

            heapq.heappush(self.open_list, best_nei)

    def calc_g(self, node_new, node_cur, shift_x, shift_y):
        """
        Calculate true cost, g
        """

        if self.field.cell_cost[node_new.x_pos_g, node_new.y_pos_g] != 1000:
            g_new = node_cur.g_val + ((shift_x / self.field.cell_size)**2 + (shift_y / self.field.cell_size)**2)**.5
        else:
            g_new = node_cur.g_val + 1000

        return g_new

    def calc_h(self, node_new):
        """
        Calculate hueristic
        """

        x_diff = abs(self.goal_loc_g[0] - node_new.x_pos_g)
        y_diff = abs(self.goal_loc_g[1] - node_new.y_pos_g)

        # Diagonal
        h_new = (x_diff + y_diff) + (1 - 2) * min(x_diff, y_diff)

        # Euclidean
        # h_new = (x_diff**2 + y_diff**2)**.5

        return h_new

    def check_for_goal(self, node):
        """
        check if node is the goal node
        """
        if(self.goal_loc_w[0] == node.x_pos_w and
           self.goal_loc_w[1] == node.y_pos_w):
            check = True
        else:
            check = False

        return check

    def plot_all_nodes(self):
        """
        Plot all nodes analyzed
        """
        x_arr = np.zeros(len(self.closed_list) + len(self.open_list) - 1)
        y_arr = np.zeros(len(self.closed_list) + len(self.open_list) - 1)
        index = 0
        # print "----------------------------"
        while index < len(self.closed_list) + len(self.open_list) - 1:
            for cl_i, node in enumerate(self.closed_list):
                x_arr[index] = node.x_pos_w
                y_arr[index] = node.y_pos_w
                index = cl_i

            for ol_i, node in enumerate(self.open_list):
                x_arr[index] = node.x_pos_w
                y_arr[index] = node.y_pos_w
                index = ol_i + len(self.closed_list)

        plt.plot(x_arr, y_arr, 'ko', markersize=1)

        self.field.fig.canvas.draw()

    def plot_final_path(self):
        """
        plot the final path
        """

        x_arr = [self.goal_node.x_pos_w]
        y_arr = [self.goal_node.y_pos_w]

        next_par_id = self.goal_node.parent_id

        self.fin_path = [self.goal_node]

        while next_par_id is not None:

            for _ig, node in enumerate(self.closed_list):

                if node.node_id == next_par_id:
                    # pprint(vars(node))
                    x_arr.append(node.x_pos_w)
                    y_arr.append(node.y_pos_w)
                    next_par_id = node.parent_id
                    self.fin_path.append(node)
                    continue

        self.fin_path.reverse()
        plt.plot(x_arr, y_arr, 'ro-', markersize=2, linewidth=2)
        self.field.fig.canvas.draw()

class Grid(object):
    """
    construct map with obstacles
    """

    def __init__(self, width, addition):

        self.cell_size = width

        self.x_axis = [-2, 5]
        self.y_axis = [-6, 6]

        self.x_lim = np.arange(self.x_axis[0], self.x_axis[1], self.cell_size)
        self.y_lim = np.arange(self.y_axis[0], self.y_axis[1], self.cell_size)

        self.marker_data = load_data('ds1_Landmark_Groundtruth.dat', 3, 0, [0, 2, 4, 6, 8])

        self.cell_cost = np.ones([len(self.x_lim), len(self.y_lim)])

        # compare landmark location to x and y values
        for _x, marker in enumerate(self.marker_data):

            x_hit_min, y_hit_min = self.get_grid_index(marker[1] - addition, marker[2] - addition)

            x_hit_max, y_hit_max = self.get_grid_index(marker[1] + addition, marker[2] + addition)

            x_hit = np.arange(x_hit_min, x_hit_max, 1)
            y_hit = np.arange(y_hit_min, y_hit_max, 1)

            if len(x_hit) == 0:
                x_hit = [x_hit_min]
            if len(y_hit) == 0:
                y_hit = [y_hit_min]

            #adjust value in cost map to identify obstables
            for _ig, mark_x in enumerate(x_hit):
                for _ig, mark_y in enumerate(y_hit):

                    self.cell_cost[mark_x][mark_y] = 1000

        # parse landmark data to plot
        _ignore, land_x, land_y, _ignore, _ignore = map(list, zip(*self.marker_data))

        # # add landmark labels
        # for _a, item in enumerate(landmark_data):
        #     plt.annotate('%s' %item[0], xy=(item[1], item[2]), xytext=(3, 3),
        #                  textcoords='offset points')

        self.fig, self.ax = plt.subplots()

        plt.plot(land_x, land_y, 'bo', markersize=3)

        plt.xticks(np.arange(self.x_axis[0], self.x_axis[1]+self.cell_size, 1))
        # plt.xticks(np.arange(self.x_axis[0], self.x_axis[1]+self.cell_size, .1))
        plt.yticks(np.arange(self.y_axis[0], self.y_axis[1]+self.cell_size, 1))
        # plt.yticks(np.arange(self.y_axis[0], self.y_axis[1]+self.cell_size, .1))

        plt.imshow(self.cell_cost.T, cmap="Purples", origin='lower', extent=[-2, 5, -6, 6])

        plt.grid(which="major", color='k', linewidth=1)
        plt.grid(which="minor", color='-k')

        plt.ylabel("World Y Position (m)")
        plt.xlabel("World X Position (m)")
        plt.axis([-2, 5, -6, 6])

        self.fig.canvas.draw()

    def get_grid_index(self, data_set_x, data_set_y):
        """
        returns the grid indecies corresponding to the input values.

        INPUT: data_set_x, data_set_y world position x and y value

        OUTPUT: x, y grid INDEX values that match the given x and y values

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
    Class to control the robot simulation
    """

    def __init__(self, start_loc, goal_loc, final_path=None, grid_obj=None):

        if final_path is None:
            self.cur_state = [start_loc[0], start_loc[0], (3*np.pi)/2]
            self.target_state = self.cur_state[:]
            self.path = []
            self.tracking = -1
            self.goal_state = goal_loc[:]
        else:
            self.cur_state = [final_path[0].x_pos_w, final_path[0].y_pos_w, (3*np.pi)/2]
            self.target_state = self.cur_state[:]
            self.goal_state = [final_path[-1].x_pos_w, final_path[-1].y_pos_w]
            self.path = final_path
            self.tracking = 0

        self.vel = [0, 0]

        self.robot_path = [self.cur_state[:]]

        self.acc_lim = [0.288, 5.579] # m/s^2, rad/sec^2

        self.th_thresh = .16 #10 degrees
        self.d_thresh = .05 #in meters

        self.k_th = .5
        self.k_d = .1
        self.dt = 0.1

        if grid_obj is None:
            self.grid = Grid(.1, .3)
        else:
            self.grid = grid_obj

        self.go_to_goal()

    def find_target(self):

        if self.tracking == -1:
            #Run analyze neighbors Function

            #check for goal

            #set 'arrived'

            pass
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
        Send a veloctiy command and update the position
        """
        goal_check = 0

        while goal_check == 0:

            prev_vel = [self.vel[0], self.vel[1]]

            x_dist = self.target_state[0] - self.cur_state[0]
            y_dist = self.target_state[1] - self.cur_state[1]

            # print self.cur_state[2], np.arctan2(y_dist, x_dist)

            delta_th = round(np.arctan2(y_dist, x_dist) - self.cur_state[2], 3)
            delta_d = round((x_dist**2 + y_dist**2)**.5, 3)

            # print self.tracking

            if delta_d <= self.d_thresh:

                goal_check = self.find_target()

                x_dist = self.target_state[0] - self.cur_state[0]
                y_dist = self.target_state[1] - self.cur_state[1]

                delta_th = round(np.arctan2(y_dist, x_dist) - self.cur_state[2], 3)
                delta_d = round((x_dist**2 + y_dist**2)**.5, 3)

            else:

                # if abs(delta_th) < self.th_thresh:
                #     self.k_d = 1
                # else:
                #     self.k_d = .05

                # print "Diff in Heading: " + str(delta_th)

                #calculate the linear veloctiy command
                eps = 0 #np.random.normal(0, .5)
                self.vel[0] = self.k_d * delta_d + eps

                if (self.vel[0] - prev_vel[0]) / self.dt > .9 * self.acc_lim[0]:
                    self.vel[0] = prev_vel[0] + .9 * self.dt * self.acc_lim[0]

                if (self.vel[0] - prev_vel[0]) / self.dt < -.9 * self.acc_lim[0]:
                    self.vel[0] = prev_vel[0] - .9 * self.dt * self.acc_lim[0]

                #remap difference in heading to turn the shortest direction
                if delta_th > np.pi:
                    delta_th = (delta_th - 2*np.pi) * -1
                elif delta_th < -np.pi:
                    delta_th = (delta_th + 2*np.pi) * -1

                #calculate the angular veloctiy command
                eps = 0# np.random.normal(0, .001)
                self.vel[1] = self.k_th * delta_th + eps

                # check if the velocity command is within the acceleration limit.
                if (self.vel[1] - prev_vel[1]) / self.dt > .9 * self.acc_lim[1]:
                    self.vel[1] = prev_vel[1] + .9 * self.dt * self.acc_lim[1]

                elif (self.vel[1] - prev_vel[1]) / self.dt < -.9 * self.acc_lim[1]:
                    self.vel[1] = prev_vel[1] - .9 * self.dt * self.acc_lim[1]

                # print self.tracking, delta_d, delta_th, self.vel[0], self.vel[1]


                # arrow_x = .001*np.cos(self.cur_state[2])
                # arrow_y = .001*np.sin(self.cur_state[2])
                # self.grid.ax.arrow(self.cur_state[0], self.cur_state[1], arrow_x, arrow_y, head_width=.1, zorder=1)
                #
                # plt.pause(.01)

                # self.grid.fig.canvas.draw()

                #send velocity command and calculate new position
                self.cur_state[0] = (self.cur_state[0] + self.vel[0] *
                                     np.cos(self.cur_state[2]) * self.dt)

                self.cur_state[1] = (self.cur_state[1] + self.vel[0] *
                                     np.sin(self.cur_state[2]) * self.dt)

                self.cur_state[2] = self.cur_state[2] + self.vel[1] * self.dt

                self.robot_path.append(self.cur_state[:])

        self.plot_robot_traj()

    def plot_robot_traj(self):
        """Plot the final robot trajectory"""

        for _ig, state in enumerate(self.robot_path):
            arrow_x = .001*np.cos(state[2])
            arrow_y = .001*np.sin(state[2])
            self.grid.ax.arrow(state[0], state[1], arrow_x, arrow_y, head_width=.1, zorder=5)

        self.grid.fig.canvas.draw()
