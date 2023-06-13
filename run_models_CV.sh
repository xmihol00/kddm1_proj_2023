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
    echo "Running CV on RF model..."
    python3 -u ./src/models/random_forest.py -cv 1>./logs/random_forest_CV.log 2>/dev/null

    echo "Running CV on NN model..."
    python3 -u ./src/models/neural_network.py -cv 1>./logs/neural_network_CV.log 2>/dev/null

    echo "Running CV on SVR model..."
    python3 -u ./src/models/support_vector_regression.py -cv 1>./logs/support_vector_regression_CV.log 2>/dev/null

    echo "Running CV on baseline model..."
    python3 -u ./src/models/baseline_linear.py -cv 1>./logs/baseline_linear_CV.log 2>/dev/null
fi

if [[ "${model_selector}" == "rf" ]]; then
    echo "Running CV on RF model..."
    python3 -u ./src/models/random_forest.py -cv 1>./logs/random_forest_CV.log 2>/dev/null
fi

if [[ "${model_selector}" == "nn" ]]; then
    echo "Running CV on NN model..."
    python3 -u ./src/models/neural_network.py -cv 1>./logs/neural_network_CV.log 2>/dev/null
fi

if [[ "${model_selector}" == "svr" ]]; then
    echo "Running CV on SVR model..."
    python3 -u ./src/models/support_vector_regression.py -cv 1>./logs/support_vector_regression_CV.log 2>/dev/null
fi

if [[ "${model_selector}" == "baseline" ]]; then
    echo "Running CV on baseline model..."
    python3 -u ./src/models/baseline_linear.py -cv 1>./logs/baseline_linear_CV.log 2>/dev/null
fi



