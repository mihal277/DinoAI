import base64
import os
import pickle
from collections import deque
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image


class GameWrapper:

    def __init__(self):
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"

        self.loss_file_path = "./objects/loss_df.csv"
        self.loss_df = pd.read_csv(self.loss_file_path) if os.path.isfile(self.loss_file_path) else pd.DataFrame(columns=['loss'])

        self.actions_file_path = "./objects/actions_df.csv"
        self.actions_df = pd.read_csv(self.actions_file_path) if os.path.isfile(self.actions_file_path) else pd.DataFrame(columns=['actions'])

        self.scores_file_path = "./objects/scores_df.csv"
        self.scores_df = pd.read_csv(self.scores_file_path) if os.path.isfile(self.scores_file_path) else pd.DataFrame(columns=['scores'])

        self.q_value_file_path = "./objects/q_values.csv"
        self.q_values_df = pd.read_csv(self.q_value_file_path) if os.path.isfile(self.q_value_file_path) else pd.DataFrame(columns=['qvalues'])

    def save_obj(self, obj, name):
        with open('objects/' + name + '.pkl', 'wb') as f:  # dump files into objects folder
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(self, name):
        with open('objects/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)

    def show_img(self, graphs=False):
        """
        Show images in new window
        """
        while True:
            screen = (yield)
            window_title = "logs" if graphs else "game_play"
            cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
            imS = cv2.resize(screen, (800, 400))
            cv2.imshow(window_title, screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    def process_img(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Grey Scale
        image = image[:300, :500]  # Crop Region of Interest(ROI)
        image = cv2.resize(image, (80, 80))
        return image

    def grab_screen(self, driver):
        image_b64 = driver.execute_script(self.getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        image = self.process_img(screen)  # processing image as required
        return image

    def init_cache(self, initial_epsilon):
        """initial variable caching, done only once"""
        self.save_obj(initial_epsilon, "epsilon")
        t = 0
        self.save_obj(t, "time")
        D = deque()
        self.save_obj(D, "D")
