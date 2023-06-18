# INP.31202UF - Knowledge Discovery and Data Mining 1
This repository contains our implementation of the **Project 4: regression** using the `Universities.csv` dataset in summer semester 2023 at TU Graz.

## Goal of our project
The goal is to the tuition fees (both for under and post graduate programs) of a university based on its ranking, location, acceptance rate, and other features. To achieve that by comparing performance of various machine learning regression models and later on creating an ensemble of all the tested models to further improve the prediction.

## Project structure
```
 |
 |-- data/                          - directory with the original and pre-processed dataset
 |
 |-- logs/                          - logs capturing training and testing of our models 
 |
 |-- plots/                         - directory with plots analyzing the dataset
 |
 |-- presentation/                  - source files for the presentation of the project
 |
 |-- results/                       - directory with CSV files containing predictions of our models with ground truths
 |
 |-- src/                           - directory with source code files for dataset pre-processing, visualizations and model training
 |
 |-- dataset_analysis.md            - markdown file containing analysis and taken pre-processing steps of the Universities.csv dataset
 |
 |-- model_training_evaluation.md   - markdown file describing how we trained and evaluated our models
 |
 |-- presentation.pdf               - presentation of the project
 |
 |-- README.md
 |
 |-- run_model.sh                   - shell script to evaluate a model on random seeds [40-49]
 |
 |-- run_models_CV.sh               - shell script to run cross validation of a selected or all models
 |
 |-- run_models_evaluation.sh       - shell script to evaluate a selected model or all models
 |
 |-- run_preprocessing.sh           - shell script to run all pre-processing steps
 |
 |-- transcript.md                  - transcript of the video presentation
```

## Result reproduction
The results can be reproduced using the provided shell scripts listed above. The execution of the shell scripts may take several hours, therefore we are already providing all the results as well as the logs capturing the training end evaluation process in the `results/` and `logs/` directories. The following examples show to execute the scripts, make sure to install all Python libraries listed in the `requirements.txt` file prior the execution:
```
./run_preprocessing.sh                             # pre-processes the data set with a default random seed of 42
export RANDOM_SEED=1; ./run_preprocessing.sh       # pre-processes the data set with a random seed of 1
``` 

```
./run_models_CV.sh linear                     # run cross-validation of the linear model on seeds [40-49]
./run_models_CV.sh nn                         # run cross-validation of the neural network model on seeds [40-49]
./run_models_CV.sh rf                         # run cross-validation of the random forest model on seeds [40-49]
./run_models_CV.sh svc                        # run cross-validation of the support vector regression model on seeds [40-49]
./run_models_CV.sh all                        # run cross-validation of all the models on seeds [40-49]
```

```
./run_models_evaluation.sh linear             # run evaluation of the linear model on seeds [40-49]
./run_models_evaluation.sh nn                 # run evaluation of the neural network model on seeds [40-49]
./run_models_evaluation.sh rf                 # run evaluation of the random forest model on seeds [40-49]
./run_models_evaluation.sh svc                # run evaluation of the support vector regression model on seeds [40-49]
./run_models_evaluation.sh ensemble           # run evaluation of the ensemble model on seeds [40-49]
./run_models_evaluation.sh all                # run evaluation of all the models including the ensemble on seeds [40-49]
```

The scripts in the `src/` directory can be of course run separately as well, see the scripts itself for expected command line arguments.