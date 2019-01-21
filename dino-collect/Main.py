import os
import time

from DataCollector import DataCollector
from Game import Game

if __name__ == "__main__":
    os.system("rm output/*")
    game = Game()
    data_collector = DataCollector(game)
    INTERVAL = 0.03
    MAX_DISTANCE = 1400  # After this dino is moving with max speed
    MAX_FRAMES = 10000
    try:
        while True:
            if game.is_crashed():
                print('Collected (valid:', data_collector.valid_counter,\
                    ', total:', data_collector.counter, ')')
                game.restart()
                if data_collector.valid_counter > MAX_FRAMES:
                    raise KeyboardInterrupt
            elif data_collector.get_dino_distance() > MAX_DISTANCE:
                game.toggle_bot()
                while not game.is_crashed():
                    time.sleep(1.)
                game.toggle_bot()
            else:                    
                data_collector.grab_screen_data()
                time.sleep(INTERVAL)
    except Exception:
        game.end()
        raise Exception
