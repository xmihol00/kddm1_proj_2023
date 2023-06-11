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

if [[ "${model_selector}" == "rf" ]]; then
    echo "generate CV log by model..."
    python3 -u ./src/models/random_forest.py 1>./logs/random_forest_CV.log 2>/dev/null
    
    echo "generate seed logs by run_model.sh ..."
    ./run_model.sh ./src/models/random_forest.py RF_predicted_truth RF
fi

if [[ "${model_selector}" == "nn" ]]; then
    echo "generate CV log by model..."
    python3 -u ./src/models/neural_network.py 1>./logs/neural_network_CV.log 2>/dev/null
    
    echo "generate seed logs by run_model.sh ..."
    ./run_model.sh ./src/models/neural_network.py NN_predicted_truth NN
fi

if [[ "${model_selector}" == "svr" ]]; then
    echo "generate CV log by model..."
    python3 -u ./src/models/support_vector_regression.py 1>./logs/support_vector_regression_CV.log 2>/dev/null
    
    echo "generate seed logs by run_model.sh ..."
    ./run_model.sh ./src/models/support_vector_regression.py SVR_predicted_truth SVR
fi

if [[ "${model_selector}" == "baseline" ]]; then
    echo "generate CV log by model..."
    python3 -u ./src/models/baseline_linear.py 1>./logs/baseline_linear_CV.log 2>/dev/null
    
    echo "generate seed logs by run_model.sh ..."
    ./run_model.sh ./src/models/baseline_linear.py baseline_predicted_truth baseline
fi



