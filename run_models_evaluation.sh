#!/usr/bin/env bash
# -*- coding: utf-8 -*-

################################################################################
### input

usage='
    prior usage: generate logs
    Usage: ./run_gen_model_logs.sh {model_selector}
    eg.:   ./run_gen_model_logs.sh rf
'

[ -z "$1" ] && echo -e "Req: arg1 missing.\n$usage" && exit 1    || model_selector="$1"

################################################################################
### code

if [[ "${model_selector}" == "all" ]]; then
    echo "evaluating RF model..."
    ./run_model.sh ./src/models/random_forest.py RF_predicted_truth RF
    
    echo "evaluating NN model..."
    ./run_model.sh ./src/models/neural_network.py NN_predicted_truth NN
    
    echo "evaluating SVR model..."
    ./run_model.sh ./src/models/support_vector_regression.py SVR_predicted_truth SVR
    
    echo "evaluating linear model..."
    ./run_model.sh ./src/models/baseline_linear.py baseline_predicted_truth baseline

    echo "evaluating ensemble model..."
    python3 ./src/models/ensemble.py 1>./logs/ensemble_average.log 
fi

if [[ "${model_selector}" == "rf" ]]; then
    echo "evaluating RF model..."
    ./run_model.sh ./src/models/random_forest.py RF_predicted_truth RF
fi

if [[ "${model_selector}" == "nn" ]]; then
    echo "evaluating NN model..."
    ./run_model.sh ./src/models/neural_network.py NN_predicted_truth NN
fi

if [[ "${model_selector}" == "svr" ]]; then
    echo "evaluating SVR model..."
    ./run_model.sh ./src/models/support_vector_regression.py SVR_predicted_truth SVR
fi

if [[ "${model_selector}" == "linear" ]]; then    
    echo "evaluating linear model..."
    ./run_model.sh ./src/models/baseline_linear.py baseline_predicted_truth baseline
fi

if [[ "${model_selector}" == "ensemble" ]]; then    
    echo "evaluating ensemble model..."
    python3 ./src/models/ensemble.py 1>./logs/ensemble_average.log 
fi



