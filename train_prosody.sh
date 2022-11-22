#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=2


python main.py  --config /mnt/data1/waris/repo/transformer-prosody-vc/conf/transformer_prosody_predictor.yaml \
                --name=prosody-predictor \
                --seed=2 \
                --prosodypredictor