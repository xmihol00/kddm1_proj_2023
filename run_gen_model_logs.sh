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
    echo "generate seed logs by run_model.sh ..."
    ./run_model.sh ./src/models/random_forest.py RF_predicted_truth RF
    
    echo "generate CV log by model..."
    python3 -u ./src/models/random_forest.py 1>./logs/random_forest_CV.txt 2>/dev/null
fi



