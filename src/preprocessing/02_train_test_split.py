import os, sys
from pathlib import Path
import pandas as pd

os.chdir(Path(__file__).parents[2])
sys.path.append(os.getcwd())
from src.utils import split, RANDOM_SEED
from src.utils import get_rel_data_path

DATA_PATH = get_rel_data_path()
os.makedirs(DATA_PATH[2], exist_ok=True)


################################################################################
#sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
print("Splitting data into train and test sets...")


################################################################################
universities = pd.read_csv(DATA_PATH[1] + "Universities_cleaned_deduplicated.csv")
universities.set_index("Id", inplace=True)

# split 80 % for training and 20 % for testing
universities_train = universities.sample(frac=0.8, random_state=RANDOM_SEED)
universities_test = universities.drop(universities_train.index)

universities_train.to_csv(DATA_PATH[2] + "Universities_train_split.csv")
universities_test.to_csv(DATA_PATH[2] + "Universities_test_split.csv")


################################################################################
# by thomas
universities = pd.read_csv(DATA_PATH[1] + "Universities_cleaned_deduplicated_by_thomas.csv")
universities = universities.drop(columns='Unnamed: 0')

# split 80 % for training and 20 % for testing
universities_train, universities_test = split(universities, test_size=0.2)
universities_train.to_csv(DATA_PATH[2] + "Universities_train_split_non_biased.csv")
universities_test.to_csv(DATA_PATH[2] + "Universities_test_split_non_biased.csv")
