import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from seed import RANDOM_SEED

print("Splitting data into train and test sets...")

universities = pd.read_csv("data/Universities_cleaned_deduplicated.csv")
universities.set_index("Id", inplace=True)

# split 80 % for training and 20 % for testing
universities_train = universities.sample(frac=0.8, random_state=RANDOM_SEED)
universities_test = universities.drop(universities_train.index)

universities_train.to_csv("data/Universities_train_split.csv")
universities_test.to_csv("data/Universities_test_split.csv")
