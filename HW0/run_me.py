"""Main Run File"""

import robot_library


def main():
    """Main Routine"""
    bot = robot_library.Robot()
    bot.part_a6()
    bot.part_a2()
    bot.part_a3()
    bot.part_b7()

    bot.create_plots()


main()
