"""Main Run File"""

import a_star


def main():
    """Main Routine"""

    # Part 3
    # path3_1 = a_star.AStar([.5, -1.5], [0.5, 1.5], 1, 0, online_check=0)
    # path3_1.start_planning()
    #
    # path3_2 = a_star.AStar([4.5, 3.5], [4.5, -1.5], 1, 0, online_check=0)
    # path3_2.start_planning()
    #
    # path3_3 = a_star.AStar([-0.5, 5.5], [1.5, -3.5], 1, 0, online_check=0)
    # path3_3.start_planning()
    #
    # # Part 5
    # part5_1 = a_star.AStar([.5, -1.5], [0.5, 1.5], 1, 0)
    # part5_1.start_planning()
    #
    # part5_2 = a_star.AStar([4.5, 3.5], [4.5, -1.5], 1, 0)
    # part5_2.start_planning()
    #
    # part5_3 = a_star.AStar([-0.5, 5.5], [1.5, -3.5], 1, 0)
    # part5_3.start_planning()
    #
    # # Part 7
    path7_1 = a_star.AStar([2.45, -3.55], [0.95, -1.55], .1, .3)
    path7_1.start_planning()
    #
    # path7_2 = a_star.AStar([4.95, -0.05], [2.45, 0.25], .1, .3)
    # path7_2.start_planning()
    #
    # path7_3 = a_star.AStar([-0.55, 1.45], [1.95, 3.95], .1, .3)
    # path7_3.start_planning()
    #
    # # Part 9
    # a_star.Robot([2.45, -3.55], [0.95, -1.55], final_path=path7_1.fin_path, grid_obj=path7_1.field)
    # a_star.Robot([4.95, -0.05], [2.45, 0.25], final_path=path7_2.fin_path, grid_obj=path7_2.field)
    # a_star.Robot([-0.55, 1.45], [1.95, 3.95], final_path=path7_3.fin_path, grid_obj=path7_3.field)

    # Part 10
    a_star.Robot([2.45, -3.55], [0.95, -1.55])
    a_star.Robot([4.95, -0.05], [2.45, 0.25])
    a_star.Robot([-0.55, 1.45], [1.95, 3.95])

    # Part 11

    a_star.plot_show()

main()
