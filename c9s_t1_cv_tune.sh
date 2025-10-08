#!/bin/bash
HPOPT_EXPERIMENT_NAME=c9s_t1_test
HPOPT_CONFIG_FILE=$(pwd)/training/config/orion-tpe.yaml
orion hunt -n $HPOPT_EXPERIMENT_NAME -c $HPOPT_CONFIG_FILE python training/train.py training/config/template.yaml \
    --lr_start~"loguniform(1e-5, 1e-3)" \
    --dp~"uniform(0.0, 0.5)" \
    --batch_size~"choices([4, 8, 16, 32])" \
    --num_fc_neurons~"choices([256, 512, 768, 1024])"