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
        self.dino_states = {}
        self.driver = game.get_driver()
        self.getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"

    def grab_screen_data(self):
        image_b64 = self.driver.execute_script(self.getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        cv2.imwrite("output/" + str(self.counter) + ".jpg", screen)
        self.dino_states[self.counter] = self.get_dino_state()
        self.save_obj(self.dino_states, "output/dino_states")
        self.counter += 1

    def get_dino_state(self):
        if self.driver.execute_script("return Runner.instance_.tRex.jumping"):
            return DinoState.JUMPING
        elif self.driver.execute_script("return Runner.instance_.tRex.ducking"):
            return DinoState.DUCKING
        return DinoState.RUNNING

    def save_obj(self, obj, name):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
