from DatasetExtractor import ImageAndSpeedDatasetExtractor
from DinoGameRunner import DinoGameRunner

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dataset Extraction Example')

    parser.add_argument("--game_path", "-p", action="store", required=True)
    parser.add_argument("--output_dir", "-o", action="store", default=".")
    args = parser.parse_args()

    game_runner = DinoGameRunner(args.game_path)
    extractor = ImageAndSpeedDatasetExtractor(game_runner, args.output_dir)
    game_runner.run_game(dataset_extractor=extractor)
