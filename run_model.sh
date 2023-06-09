#!/bin/bash

# Usage:
# run_model.sh <path/model_name.py, e.g. src/models/neural_network.py> <start of names of files in results/, e.g. NN_predicted_truth> <model name, e.g. "Neural network">
# e.g.: ./run_model.sh ./src/models/random_forest.py RF_predicted_truth RF

for i in {0..4}
do
    echo "Running with RANDOM_SEED=$i ..."
    export RANDOM_SEED=$i
    # preprocess data set with the corresponding seed
    ./run_preprocessing.sh
    # run a model with the corresponding seed (python3 -u <path/model_name.py> >./logs/<model_name>_seed<seed>.txt)
    echo "Running model $3 ..."
    python3 -u $1 1>./logs/$(basename ${1%.*})_seed$i.txt 2>/dev/null
    echo ""
done

echo "Averaging model performance ..."
python3 src/average_model_performance.py $2 "$3" 1>./logs/$(basename ${1%.*})_average.txt

# reset the pre-processed data set
export RANDOM_SEED=14
./run_preprocessing.sh
