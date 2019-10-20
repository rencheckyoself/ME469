"""Main Run File"""

import a_star


def main():
    """Main Routine"""

    # pos_set_one = a_star.AStar([.5, -1.5], [0.5, 1.5], 1, 0)
    # pos_set_two = a_star.AStar([4.5, 3.5], [4.5, -1.5], 1, 0)
    # pos_set_three = a_star.AStar([-0.5, 5.5], [1.5, -3.5], 1, 0)

    pos_set_four = a_star.AStar([2.45, -3.55], [0.95, -1.55], .1, .3)
    # pos_set_five = a_star.AStar([4.95, -0.5], [2.45, 0.25], 1)
    # pos_set_six = a_star.AStar([-0.55, 1.45], [1.95, 3.95], 1)

    pos_set_four.plot_show()

main()
