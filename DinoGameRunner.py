from io import BytesIO
import time

from PIL import Image
from selenium.webdriver import Firefox, FirefoxProfile
from selenium.webdriver import ActionChains
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.options import Options

import numpy as np
import base64
from io import BytesIO

import cv2
from PIL import Image


WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800


class DinoState:
    RUNNING = "running"
    JUMPING = "jumping"
    DUCKING = "ducking"


class DinoGameRunner:
    def __init__(self, game_path):
        fp = FirefoxProfile()

        fp.add_extension(extension='uBlock0_1.17.7b1.firefox.signed.xpi')
        self.driver = Firefox(firefox_profile=fp)
        self.driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        self.game_path = game_path

    def _get_dino_distance(self):
        return self.driver.execute_script(
            "return Runner.instance_.distanceMeter"
            ".getActualDistance(Runner.instance_.distanceRan)"
        )

    def _get_dino_state(self):
        if self.driver.execute_script("return Runner.instance_.tRex.jumping"):
            return DinoState.JUMPING
        elif self.driver.execute_script("return Runner.instance_.tRex.ducking"):
            return DinoState.DUCKING
        return DinoState.RUNNING

    def _get_dino_speed(self):
        return self.driver.execute_script(
            "return Runner.instance_.currentSpeed"
        )

    def _dino_crashed(self):
        return self.driver.execute_script("return Runner.instance_.crashed")

    def _dino_stop(self):
        self.driver.execute_script("Runner.instance_.stop()")

    def _dino_start(self):
        self.driver.execute_script("Runner.instance_.play()")

    def _become_immortal(self):
        self.driver.execute_script("Runner.prototype.gameOver = function(){}")

    def _get_dino_params(self):
        return {
            "distance": self._get_dino_distance(),
            "speed": self._get_dino_speed(),
            "dino_state": self._get_dino_state(),
        }

    def _make_initial_jump(self):
        self.press_key("space")

    def take_screenshot(self):
        getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); return canvasRunner.toDataURL().substring(22)"
        image_b64 = self.driver.execute_script(getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        return screen

    def press_key(self, key="space"):
        key = {
            "space": Keys.SPACE,
            "arrow_down": Keys.ARROW_DOWN
        }[key]
        ActionChains(self.driver).key_down(key).perform()

    def _make_background_white(self):
        self.driver.execute_script("document.body.style.backgroundColor = 'white';")

    def _set_activate_bot(self):
        self.driver.find_element_by_id("botStatus").click()

    def run_game(self,
                 ai=None,
                 dataset_extractor=None,
                 exit_on_crash=True,
                 make_initial_jump=True,
                 immortal=False):
        self.driver.get(self.game_path)

        self._make_background_white()
        self._set_activate_bot()

        if immortal:
            self._become_immortal()

        if make_initial_jump:
            self._make_initial_jump()

        while not self._dino_crashed():

            # self._dino_stop()
            dino_params = self._get_dino_params()

            if ai:
                ai.perform_action(dino_params)
                # time.sleep(0.01)

            if dataset_extractor:
                dataset_extractor.update_dino_params(dino_params)
            # self._dino_start()

        if dataset_extractor:
            dataset_extractor.dump_dataset()

        print(self._get_dino_params())

        if exit_on_crash:
            self.exit()

    def exit(self):
        self.driver.close()

class FreezingGameRunner:
    KEYMAP = {
        'space': Keys.SPACE,
        'arrown_down': Keys.ARROW_DOWN
    }

    def __init__(self, game_path, headless=False):
        self._game_path = game_path
        self._options = Options()
        if headless:
            self._options.add_argument('--headless')
        self._driver = None

    def _make_initial_jump(self):
        self.press_key("space")

    def _initiate_driver(self):
        self._driver = Firefox(firefox_options=self._options)
        self._driver.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)

    def _game_resume(self):
        self._driver.execute_script("Runner.instance_.play()")

    def _game_pause(self):
        self._driver.execute_script("Runner.instance_.stop()")

    def press_key(self, key="space"):
        if key in FreezingGameRunner.KEYMAP:
            k = FreezingGameRunner.KEYMAP[key]
            ActionChains(self._driver).key_down(k).perform()

    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_score(self):
        digits = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits"
        )
        if not digits:
            return 0
        return int(''.join(digits))

    def take_screenshot(self):
        canvas = self._driver.find_element_by_class_name("runner-canvas")
        location = canvas.location_once_scrolled_into_view
        size = canvas.size
        screenshot = self._driver.get_screenshot_as_png()
        im = Image.open(BytesIO(screenshot))
        left = location['x']
        top = location['y']
        right = location['x'] + size['width']
        bottom = location['y'] + size['height']
        return im.crop((left, top, right, bottom))

    def start(self, seconds):
        self._initiate_driver()
        self._driver.get(self._game_path)

        self._make_initial_jump()

        if seconds > 0:
            time.sleep(seconds)

        if not self.is_crashed():
            self._game_pause()

    def play(self, seconds):
        if not self.is_crashed():
            self._game_resume()
            if seconds > 0:
                time.sleep(seconds)
            if not self.is_crashed():
                self._game_pause()

    def exit(self):
        self._driver.close()
