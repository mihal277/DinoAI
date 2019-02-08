from collect.collect_raw_output import collect_output
from collect.raw_output_to_dataset import make_four_pictures_dataset

MAX_FILE_LEN = 300
OUTPUT_PATH = './output'

if __name__ == '__main__':
    collect_output()
    make_four_pictures_dataset(OUTPUT_PATH, MAX_FILE_LEN)
