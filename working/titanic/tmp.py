def HELLO():
    print("Hello.")

import os

from titanic import dataset

input_dir = os.path.join("..", "..", "input", "titanic")
dataset.read_csv_file(os.path.join(input_dir, "train.csv"))