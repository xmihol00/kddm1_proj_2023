import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # ensure includes of our files work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from constants import RANDOM_SEED, DATA_PATH

if __name__ == "__main__":
    print("Splitting data into train and test sets...")
    os.makedirs(DATA_PATH["split"], exist_ok=True) # ensure output directory exists

    universities = pd.read_csv(DATA_PATH["cleaning"] + "Universities_cleaned_deduplicated.csv")
    universities.set_index("Id", inplace=True)

    # split 80 % for training and 20 % for testing
    universities_train = universities.sample(frac=0.8, random_state=RANDOM_SEED)
    universities_test = universities.drop(universities_train.index) # do not include the "Id" column

    universities_train.to_csv(DATA_PATH["split"] + "Universities_train_split.csv")
    universities_test.to_csv(DATA_PATH["split"] + "Universities_test_split.csv")
