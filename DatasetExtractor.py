from DinoGameRunner import DinoState

import numpy as np

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
        self.old_dino_params = self.current_dino_params
        self.current_dino_params = dino_params
        if not self._same_distance():
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

    def _same_distance(self):
        old_distance = self.old_dino_params.get("distance")
        current_distance = self.current_dino_params.get("distance")
        return old_distance == current_distance

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
        self.dataset_x.append(screenshot)

        action = self._determine_action()
        self.dataset_y.append(action)

        self.speeds.append(self.current_dino_params["speed"])

    def dump_dataset(self):
        super().dump_dataset()
        self.speeds = np.array(self.speeds)
        np.save(self.directory / "speeds.npy", self.dataset_x)





