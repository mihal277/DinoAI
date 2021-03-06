from DatasetExtractor import MultipleImagesDatasetExtractor
from DinoGameRunner import DinoGameRunner

from keras.models import load_model
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import numpy as np
import cv2

import argparse
import os
from collections import deque

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

class ConvNetPlayer(object):
    def __init__(self, model_path, runner):
        self.model = load_model(model_path, custom_objects={'auroc': auroc})
        self.runner = runner
        self.last_distance = 0

    def preprocess_im(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = image[:300, :500]
        image = cv2.resize(image, (80,80))
        # cv2.imwrite("test2.png", image)
        return image

    def perform_action(self, dino_params):
        if dino_params["distance"] <= self.last_distance:
            return
        self.last_distance = dino_params["distance"]
        picture = self.preprocess_im(self.runner.take_screenshot())
        
        action = self.model.predict(
            [picture.reshape((1,80,80,1)), np.array([dino_params["speed"]]).reshape(1,1)]
        
        # print(action)
        ).argmax()

        if dino_params["distance"] < 42:
            return
        if action == 1:
            self.runner.press_key("space")
        

class ConvNet4FramePlayer:
    def __init__(self, model_path, runner):
        self.model = load_model(model_path, custom_objects={'auroc': auroc})
        self.runner = runner
        self.last_distance = 0
        self.frame_queue = deque()

    def preprocess_im(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        image = image[:300, :500]
        image = cv2.resize(image, (80,80))
        # cv2.imwrite("test2.png", image)
        return image

    def perform_action(self, dino_params):
        if dino_params['distance'] <= self.last_distance:
            return
        self.last_distance = dino_params['distance']
        picture = self.preprocess_im(self.runner.take_screenshot())
        self.frame_queue.append(picture)
        if len(self.frame_queue) < 4:
            return
        elif len(self.frame_queue) == 5:
            self.frame_queue.popleft()
        if dino_params['distance'] < 42:
            return
        if dino_params['dino_state'] == 'jumping':
            return
        action = self.model.predict(np.array(self.frame_queue).reshape((1, 80, 80, 4)))[0]
        print('action', action, action.argmax())
        if action.argmax() == 1:
            self.runner.press_key('space')


if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(__file__))
    # t_rex_runner_url = "file://" + working_dir + "/t-rex-runner/index.html"
    t_rex_runner_url = "https://elgoog.im/t-rex/"

    parser = argparse.ArgumentParser(description='AI playing')

    parser.add_argument("--game_path", "-p", action="store", default=t_rex_runner_url)
    parser.add_argument("--model_path", "-m", action="store", default="with_speeds.h5")
    parser.add_argument("--four_frames", "-4", action="store_true", default=False)
    args = parser.parse_args()

    game_runner = DinoGameRunner(args.game_path)
    if args.four_frames:
        ai = ConvNet4FramePlayer(args.model_path, game_runner)
    else:
        ai = ConvNetPlayer(args.model_path, game_runner)
    game_runner.run_game(ai=ai)
 
