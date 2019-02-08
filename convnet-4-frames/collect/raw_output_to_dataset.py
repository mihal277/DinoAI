import argparse
from collections import deque
from pathlib import Path
import pickle

import cv2
import numpy as np

from collect.DataCollector import DinoState

DINO_STATES_FN = "dino_states.pkl"


class Actions:
    NOTHING = 0
    JUMP = 1
    DUCK = 2
    ABORT = 3


def get_dino_states(output_dir):
    with (output_dir / DINO_STATES_FN).open("rb") as f:
        dino_states = pickle.load(f)
    return dino_states


def add_screenshot_to_list(screenshot, l, n=4):
    if len(l) == n:
        l.popleft()
    l.append(screenshot)


def dump_dataset(name, output_dir, X, y, x_shape):
    X, y = np.array(X).reshape(x_shape), np.array(y)
    np.savez(output_dir / f"X_{name}.npz", X)
    np.savez(output_dir / f"y_{name}.npz", y)


def determine_action(i, dino_states):
    curr_state = dino_states[i]["dino_state"]
    next_state = dino_states[i+1]["dino_state"]
    if curr_state == DinoState.RUNNING and next_state == DinoState.JUMPING:
        return Actions.JUMP
    if curr_state == DinoState.DUCKING or next_state == DinoState.DUCKING:
        return Actions.DUCK
    if curr_state == DinoState.JUMPING:
        return Actions.ABORT
    return Actions.NOTHING


def make_four_pictures_dataset(output_path, max_file_len):
    output_dir = Path(output_path)
    dino_states = get_dino_states(output_dir)
    images_count = len([f for f in output_dir.iterdir() if f.suffix == ".png"])
    X, y = [], []
    output_file_idx = 0
    last_screenshots = deque()
    for i in range(images_count - 2):
        if len(y) == max_file_len:
            dump_dataset(str(output_file_idx), output_dir, X, y,
                         (len(X), 4, X[0][0].shape[0], X[0][0].shape[1]))
            output_file_idx += 1
            X, y = [], []
        im = cv2.imread(str(output_dir / f"{str(i)}.png"), 0)
        add_screenshot_to_list(im, last_screenshots)
        if len(last_screenshots) == 4:
            action = determine_action(i, dino_states)
            if action != Actions.ABORT:
                X.append(
                    np.array([screenshot for screenshot in last_screenshots])
                )
                y.append(action)
    if len(y) and len(last_screenshots) == 4:
        dump_dataset("f" + str(output_file_idx), output_dir, X, y, (len(X), 4, X[0][0].shape[0], X[0][0].shape[1]))
