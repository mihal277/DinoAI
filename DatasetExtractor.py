from DinoGameRunner import DinoState

import numpy as np

from collections import deque
from pathlib import Path


class Actions:
    NOTHING = 0
    JUMP = 1
    DUCK = 2


class DatasetExtractor:
    def __init__(self, runner, directory):
        self.runner = runner
        self.old_dino_params = {}
        self.current_dino_params = {}
        self.dataset_x = []
        self.dataset_y = []
        self.directory = Path(directory)

    def update_dino_params(self, dino_params):
        if self._same_distance(dino_params):
            return
        self.old_dino_params = self.current_dino_params
        self.current_dino_params = dino_params
        self.update_dataset()

    def _determine_action(self):
        old_state = self.old_dino_params.get("dino_state")
        current_state = self.current_dino_params.get("dino_state")
        if old_state == DinoState.RUNNING:
            if current_state == DinoState.JUMPING:
                return Actions.JUMP
            elif current_state == DinoState.DUCKING:
                return Actions.DUCK
        return Actions.NOTHING

    def _same_distance(self, new_params):
        old_distance = self.current_dino_params.get("distance")
        new_distance = new_params.get("distance")
        return old_distance == new_distance

    def dump_dataset(self):
        self.dataset_x = np.array([np.array(img) for img in self.dataset_x])
        self.dataset_y = np.array(self.dataset_y)
        np.save(self.directory / "x.npy", self.dataset_x)
        np.save(self.directory / "y.npy", self.dataset_y)

    def update_dataset(self):
        raise NotImplementedError()


class ImageAndSpeedDatasetExtractor(DatasetExtractor):
    def __init__(self, runner, directory):
        super().__init__(runner, directory)
        self.speeds = []

    def update_dataset(self):
        screenshot = self.runner.take_screenshot()
        self.dataset_x.append(screenshot.convert("L"))

        action = self._determine_action()
        self.dataset_y.append(action)

        self.speeds.append(self.current_dino_params["speed"])

    def dump_dataset(self):
        super().dump_dataset()
        self.speeds = np.array(self.speeds)
        np.save(self.directory / "speeds.npy", self.dataset_x)


class MultipleImagesDatasetExtractor(DatasetExtractor):
    def __init__(self, runner, directory, n_images):
        super().__init__(runner, directory)
        self.last_screenshots = deque()
        self.n_images = n_images

    def add_screenshot(self, screenshot):
        if len(self.last_screenshots) == self.n_images:
            self.last_screenshots.popleft()
        self.last_screenshots.append(screenshot)

    def update_dataset(self):
        screenshot = self.runner.take_screenshot()
        self.add_screenshot(screenshot)

        if len(self.last_screenshots) < self.n_images:
            return

        self.dataset_x.append(
            [screenshot.convert("L") for screenshot in self.last_screenshots]
        )

        action = self._determine_action()
        self.dataset_y.append(action)

    def dump_dataset(self):
        self.dataset_x = np.array(
            [np.array([np.array(img) for img in imgs_set])
             for imgs_set in self.dataset_x]
        )
        self.dataset_y = np.array(self.dataset_y)
        np.save(self.directory / "x.npy", self.dataset_x)
        np.save(self.directory / "y.npy", self.dataset_y)

