import base64
import pickle
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


class DinoState:
    RUNNING = 0
    JUMPING = 1
    DUCKING = 2


class DataCollector:

    def __init__(self, game):
        self.counter = 0
        self.valid_counter = 0
        self.previous_state = DinoState.RUNNING
        self.dino_states = {}
        self.driver = game.get_driver()
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"

    def grab_screen_data(self):
        image_b64 = self.driver.execute_script(self.getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        cv2.imwrite("output/" + str(self.counter) + ".png", screen)
        self.dino_states[self.counter] = self.get_dino_params()
        self.save_obj(self.dino_states, "output/dino_states")
        if self.previous_state != DinoState.JUMPING:
            self.valid_counter += 1
        self.previous_state = self.dino_states[self.counter]['dino_state']
        self.counter += 1

    def get_dino_distance(self):
        return self.driver.execute_script(
            "return Runner.instance_.distanceMeter"
            ".getActualDistance(Runner.instance_.distanceRan)"
        )

    def get_dino_speed(self):
        return self.driver.execute_script(
            "return Runner.instance_.currentSpeed"
        )

    def get_dino_params(self):
        return {
            "distance": self.get_dino_distance(),
            "speed": self.get_dino_speed(),
            "dino_state": self.get_dino_state(),
        }

    def get_dino_state(self):
        if self.driver.execute_script("return Runner.instance_.tRex.jumping"):
            return DinoState.JUMPING
        elif self.driver.execute_script("return Runner.instance_.tRex.ducking"):
            return DinoState.DUCKING
        return DinoState.RUNNING

    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
