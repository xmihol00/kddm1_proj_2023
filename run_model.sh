#!/bin/bash

# Usage:
# run_model.sh <path/model_name.py, e.g. src/models/neural_network.py> <start of names of files in results/, e.g. NN_predicted_truth> <model name, e.g. "Neural network">

for i in {0..4}
do
    echo "Running with RANDOM_SEED=$i..."
    export RANDOM_SEED=$i
    # preprocess data set with the corresponding seed
    ./run_preprocessing.sh
    # run a model with the corresponding seed (python3 -u <path/model_name.py> >./logs/<model_name>_seed<seed>.log)
    echo "Running model $3..."
    python3 -u $1 1>./logs/$(basename ${1%.*})_seed$i.log 2>/dev/null
    echo ""
done

# TODO: average the performance (MSE, MAE, RMSE, R2 score...) of the given model and store it in results/<model_name>_average.txt, 
# use $1, $2 and $3 (or add other parameters) to call a python script to do that
