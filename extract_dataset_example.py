from DatasetExtractor import ImageAndSpeedDatasetExtractor
from DinoGameRunner import DinoGameRunner

import argparse
import os

if __name__ == "__main__":
    working_dir = os.path.dirname(os.path.realpath(__file__))
    t_rex_runner_url = "file://" + working_dir + "/t-rex-runner/index.html"

    parser = argparse.ArgumentParser(description='Dataset Extraction Example')

    parser.add_argument("--game_path", "-p", action="store", default=t_rex_runner_url)
    parser.add_argument("--output_dir", "-o", action="store", default=".")
    args = parser.parse_args()


    game_runner = DinoGameRunner(args.game_path)
    extractor = ImageAndSpeedDatasetExtractor(game_runner, args.output_dir)
    game_runner.run_game(dataset_extractor=extractor)
