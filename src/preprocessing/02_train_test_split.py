import os
import sys
from pathlib import Path
import pandas as pd

from src.utils import split
from src.utils import RANDOM_SEED

################################################################################
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(Path(__file__).parents[2])
print("Splitting data into train and test sets...")


################################################################################
universities = pd.read_csv("data/Universities_cleaned_deduplicated.csv")
universities.set_index("Id", inplace=True)

# split 80 % for training and 20 % for testing
universities_train = universities.sample(frac=0.8, random_state=RANDOM_SEED)
universities_test = universities.drop(universities_train.index)

universities_train.to_csv("data/Universities_train_split.csv")
universities_test.to_csv("data/Universities_test_split.csv")


################################################################################
# by thomas
universities = pd.read_csv("data/Universities_cleaned_deduplicated_new.csv")
universities = universities.drop(columns='Unnamed: 0')

# split 80 % for training and 20 % for testing
universities_train, universities_test = split(universities, test_size=0.2)
universities_train.to_csv("data/Universities_train_split_non_biased.csv")
universities_test.to_csv("data/Universities_test_split_non_biased.csv")
