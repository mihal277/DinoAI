import os
import time

from DataCollector import DataCollector
from Game import Game

if __name__ == "__main__":
    os.system("rm output/*")
    game = Game()
    data_collector = DataCollector(game)
    INTERVAL = 0.03
    try:
        while True:
            if game.is_crashed():
                game.restart()
            else:
                data_collector.grab_screen_data()
                time.sleep(INTERVAL)
    except Exception:
        game.end()
        raise Exception
