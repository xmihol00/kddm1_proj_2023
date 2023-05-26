# INP.31202UF - Knowledge Discovery and Data Mining 1
This repository contains our implementation of the **Project 4: regression** using the Universities.csv dataset in summer semester 2023 at TU Graz.

## Goal of our project
The goal is to the tuition fees (both for under and post graduate programs) of a university based on its ranking, location, acceptance rate, and other features. To achieve that by comparing performance of various machine learning regression models and later on creating an ensemble of all the tested models to further improve the prediction.

## Project structure
```
 |
 |-- data/                          - directory with the original and pre-processed dataset (must be created and the dataset inserted)
 |
 |-- plots/                         - directory with plots analyzing the dataset
 |
 |-- results/                       - directory with CSV files containing predictions and ground truths of our models
 |
 |-- src/                           - directory with Python source files for dataset pre-processing, visualizations and model training
 |
 |-- dataset_analysis.md            - markdown file containing analysis and taken pre-processing steps of the Universities.csv dataset
 |
 |-- model_training_evaluation.md   - markdown file describing how we trained and evaluated our models
 |
 |-- run_preprocessing.sh           - shell script to run all pre-processing steps

```
