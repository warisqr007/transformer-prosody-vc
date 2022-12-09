#!/bin/bash

. ./path.sh || exit 1;
export CUDA_VISIBLE_DEVICES=1


python main.py  --config /mnt/data1/waris/repo/transformer-prosody-vc/conf/transformer_prosody_predictor.yaml \
                --name=prosody-predictor-mel-attn \
                --seed=2 \
                --prosodypredictor
